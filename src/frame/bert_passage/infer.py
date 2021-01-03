# -*- coding: utf-8 -*-
import sys
import os
from os.path import join, dirname, abspath, exists

sys_path = dirname(dirname(abspath(__file__)))
parent_sys_path = dirname(sys_path)

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

import io
import shutil
import copy
from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils.config_loader as config
from utils.config_loader import logger, path_parser, config_meta, meta_model_name, config_model
import utils.tools as tools
import tools.general_tools as general_tools
import frame.ir.ir_tools as ir_tools
import frame.bert_qa.qa_config as qa_config
import frame.bert_passage.data_pipe_cluster as data_pipe_cluster
from frame.bert_passage.passage_obj import pid2obj
import summ.rank_sent as rank_sent
import summ.select_sent as select_sent
import summ.compute_rouge as rouge


if config.meta_model_name != 'bert_passage':
    raise ValueError('Invalid meta_model_name: {}'.format(config.meta_model_name))

if not config.grain.startswith('passage'):
    raise ValueError('Invalid grain: {}'.format(config.grain))

# token_logits_dp = join(path_parser.graph_token_logits, qa_config.RELEVANCE_SCORE_DIR_NAME)
rel_scores_dp = join(path_parser.graph_rel_scores, qa_config.RELEVANCE_SCORE_DIR_NAME)
rank_dp = join(path_parser.summary_rank, qa_config.QA_MODEL_NAME_BERT)
ir_rec_dp = join(path_parser.summary_rank, qa_config.IR_RECORDS_DIR_NAME)

use_tdqfs = 'tdqfs' in qa_config.IR_RECORDS_DIR_NAME

if use_tdqfs:
    sentence_dp = path_parser.data_tdqfs_sentences
    query_fp = path_parser.data_tdqfs_queries
    tdqfs_summary_target_dp = path_parser.data_tdqfs_summary_targets

    test_cid_query_dicts = general_tools.build_tdqfs_cid_query_dicts(query_fp=query_fp, proc=True)
    cids = [cq_dict['cid'] for cq_dict in test_cid_query_dicts]
else:
    cids = tools.get_test_cc_ids()


def init():
    # parse args
    parser = ArgumentParser()
    parser.add_argument('n_devices',
                        nargs='?',
                        default=4,
                        help='num of devices on which model will be running on')

    args = parser.parse_args()
    all_device_ids = [0, 1, 2, 3]
    device = all_device_ids[:int(args.n_devices)]
    config_meta['device'] = device

    if not torch.cuda.is_available():
        placement = 'cpu'
        logger.info('[MAIN INIT] path mode: {0}, placement: {1}'.format(config.path_type, placement))
    else:
        if len(device) == 1:
            placement = 'single'
            torch.cuda.set_device(device[0])
        elif config_meta['auto_parallel']:
            placement = 'auto'
        else:
            placement = 'manual'

        logger.info(
            '[MAIN INIT] path mode: {0}, placement: {1}, n_devices: {2}'.format(config.path_type, placement,
                                                                                args.n_devices))
    config_meta['placement'] = placement


def _place_model(model):
    if config_meta['placement'] == 'auto':
        model = nn.DataParallel(model, device_ids=config_meta['device'])
        logger.info('[place_model] Parallel Data to devices: {}'.format(config_meta['device']))

    if config_meta['placement'] in ('auto', 'single'):
        model.cuda()

    model.eval()
    return model


def _infer_sent_score(start, end, start_logit_vec, end_logit_vec):
    """
    :param start:
    :param end:
    :param start_logit_vec:
    :param end_logit_vec:
    :return:
    """
    sent_start_logit = np.exp(start_logit_vec[start:end]).reshape(-1, 1)  # n * 1
    sent_end_logit = np.exp(end_logit_vec[start:end]).reshape(1, -1)  # n * 1

    score_mat = np.matmul(sent_start_logit, sent_end_logit)  # n * n
    score_mat = np.triu(score_mat, k=0)  # mask out elements where i < j
    score = np.max(score_mat)
    return score


def _infer_sent_score_by_batch(seg_indices, start_logits, end_logits, passage_ids_batch):
    """
        Infer and gather scores for all sentences in a batch (i.e., multiple passages).

    :param seg_indices: d_batch * ns * 2
    :param start_logits: d_batch * max_nt
    :param end_logits: d_batch * max_nt
    :return:
    """
    # iter over passages
    batch_size = len(seg_indices)
    if not batch_size == len(start_logits) == len(end_logits) == len(passage_ids_batch):
        raise ValueError(
            'Incompatible size: seg_indices: {}, start_logits: {}, end_logits: {}, passage_ids_batch: {}'.format(
                batch_size,
                start_logits.shape,
                end_logits.shape,
                passage_ids_batch.shape))

    pid2rel_scores = {}

    for sample_idx in range(batch_size):
        seg_indices_passage = seg_indices[sample_idx]
        start_logit_vec = start_logits[sample_idx]
        end_logit_vec = end_logits[sample_idx]
        pid = passage_ids_batch[sample_idx]

        rel_scores_passage = []
        for start_idx, end_idx in seg_indices_passage:  # for sentences in each passage
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            # logger.info('start_idx: {}, end_idx: {}'.format(start_idx, end_idx))
            if end_idx == -1:  # less than ns_para
                break
            score = _infer_sent_score(start=start_idx,
                                      end=end_idx,
                                      start_logit_vec=start_logit_vec,
                                      end_logit_vec=end_logit_vec)
            rel_scores_passage.append(score)

        if len(rel_scores_passage) != config_model['ns_passage']:
            logger.info('#sents in {}: {}'.format(pid, len(rel_scores_passage)))

        pid2rel_scores[pid] = np.array(rel_scores_passage)  # todo: debug this hashing

    return pid2rel_scores


def _dump(model, cluster_loader, dump_dp):
    passage_ids = cluster_loader.passage_ids

    passage_idx_start = 0
    pid2rel_scores = {}
    for _, batch in enumerate(cluster_loader):  # each batch consists of multiple passages
        feed_dict = copy.deepcopy(batch)
        del feed_dict['seg_indices']

        for (k, v) in feed_dict.items():
            with torch.no_grad():
                feed_dict[k] = Variable(v, requires_grad=False)

        batch_size = len(feed_dict['token_ids'])

        passage_idx_end = passage_idx_start + batch_size
        passage_ids_batch = passage_ids[passage_idx_start:passage_idx_end]

        output = model(feed_dict['token_ids'],
                       feed_dict['seg_ids'],
                       feed_dict['token_masks'])

        start_logits, end_logits = output
        start_logits = start_logits.cpu().detach().numpy()  # d_batch * max_nt
        end_logits = end_logits.cpu().detach().numpy()  # d_batch * max_nt
        seg_indices = batch['seg_indices'].cpu().detach().numpy()  # d_batch * * ns * 2

        pid2rel_scores_batch = _infer_sent_score_by_batch(seg_indices, start_logits, end_logits, passage_ids_batch)

        pid2rel_scores = {
            **pid2rel_scores,
            **pid2rel_scores_batch,
        }

        passage_idx_start = passage_idx_end

    dump_fp = join(dump_dp, cluster_loader.cid)
    tools.save_obj(obj=pid2rel_scores, fp=dump_fp)
    logger.info('[_dump] dumping pid2rel_scores to: {0}'.format(dump_fp))


def _get_data_loader_gen(ir_rec_dp):
    if use_tdqfs:    
        data_gen = data_pipe_cluster.TdqfsQSDataLoader(test_cid_query_dicts=test_cid_query_dicts, 
            retrieve_dp=ir_rec_dp)
        return data_gen
    
    loader_params = {
        'tokenize_narr': False,
        'query_type': qa_config.QUERY_TYPE,
        'retrieve_dp': ir_rec_dp,
    }
    data_gen = data_pipe_cluster.QSDataLoader(**loader_params)
    return data_gen


def dump_pid2rel_scores():
    model = _place_model(model=config.bert_model)
    data_loader_generator = _get_data_loader_gen(ir_rec_dp)

    if exists(rel_scores_dp):
        raise ValueError('rel_scores_dp exists: {}'.format(rel_scores_dp))
    os.mkdir(rel_scores_dp)

    for cluster_loader in data_loader_generator:
        _dump(model, cluster_loader=cluster_loader, dump_dp=rel_scores_dp)


def _load_sent_info(cid, rel_scores_dp):
    rel_scores_fp = join(rel_scores_dp, cid)
    pid2rel_scores = tools.load_obj(rel_scores_fp)

    sid2rel_scores = {}
    sid2original_sent = {}

    for pid, rel_scores in pid2rel_scores.items():
        sent_objs = pid2obj(cid=cid, pid=pid, use_tdqfs=use_tdqfs).sent_objs
        print('pid: {}, rel_scores: {}'.format(pid, rel_scores))

        if len(sent_objs) != len(rel_scores):
            raise ValueError('Incompatible size: sent_objs: {} and rel_scores: {}'.format(len(sent_objs),
                                                                                          len(rel_scores)))

        for so, score in zip(sent_objs, rel_scores):
            if so.sid not in sid2rel_scores:
                sid2rel_scores[so.sid] = score
                sid2original_sent[so.sid] = so.original_sent
                continue

            sid2rel_scores[so.sid] = max(sid2rel_scores[so.sid], score)  # choose the max one

    return sid2rel_scores, sid2original_sent


def rel_scores2rank():
    if exists(rank_dp):
        raise ValueError(f'rank_dp exists: {rank_dp}')
    os.mkdir(rank_dp)

    for cid in tqdm(cids):
        sid2rel_scores, sid2original_sent = _load_sent_info(cid=cid, rel_scores_dp=rel_scores_dp)

        sid_score_list = sorted(sid2rel_scores.items(), key=lambda item: item[1], reverse=True)
        original_sents = [sid2original_sent[sid] for sid, _ in sid_score_list]

        rank_records = rank_sent.get_rank_records(sid_score_list, sents=original_sents, flat_sents=True)

        n_sents = rank_sent.dump_rank_records(rank_records=rank_records, out_fp=join(rank_dp, cid), with_rank_idx=False)
        logger.info('Dump {} ranking records'.format(n_sents))


def qa_rank2records():
    qa_rec_dp = join(path_parser.summary_rank, qa_config.QA_RECORD_DIR_NAME_BERT)

    if exists(qa_rec_dp):
        raise ValueError('qa_rec_dp exists: {}'.format(qa_rec_dp))
    os.mkdir(qa_rec_dp)

    for cid in tqdm(cids):
        retrieval_params = {
            'model_name': qa_config.QA_MODEL_NAME_BERT,
            'cid': cid,
            'filter_var': qa_config.FILTER_VAR,
            'filter': qa_config.FILTER,
            'deduplicate': None,
        }

        retrieved_items = ir_tools.retrieve(**retrieval_params)
        ir_tools.dump_retrieval(fp=join(qa_rec_dp, cid), retrieved_items=retrieved_items)


def qa_rank2records_in_batch():
    if qa_config.FILTER == 'conf':
        filter_var_range = np.arange(0.05, 1.05, 0.05)
    else:  # topK
        interval = 10
        if qa_config.ir_config.FILTER == 'topK':
            start = interval
            end = qa_config.ir_config.FILTER_VAR + interval
        else:
            start = 40
            end = 150 + interval
        filter_var_range = range(start, end, interval)

    for filter_var in tqdm(filter_var_range):
        qa_rec_dn = qa_config.QA_RECORD_DIR_NAME_PATTERN.format(qa_config.QA_MODEL_NAME_BERT,
                                                                filter_var,
                                                                qa_config.FILTER)
        qa_rec_dp = join(path_parser.summary_rank, qa_rec_dn)

        if exists(qa_rec_dp):
            raise ValueError('qa_rec_dp exists: {}'.format(qa_rec_dp))
        os.mkdir(qa_rec_dp)

        for cid in cids:
            retrieval_params = {
                'model_name': qa_config.QA_MODEL_NAME_BERT,
                'cid': cid,
                'filter_var': filter_var,
                'filter': qa_config.FILTER,
                'deduplicate': None,
            }

            retrieved_items = ir_tools.retrieve(**retrieval_params)
            ir_tools.dump_retrieval(fp=join(qa_rec_dp, cid), retrieved_items=retrieved_items)


def tune():
    """
        Tune QA confidence / compression rate / topK
        based on Recall Rouge 2.
    :return:
    """
    if qa_config.FILTER == 'conf':
        tune_range = np.arange(0.05, 1.05, 0.05)
    else:  # topK
        interval = 10
        if qa_config.ir_config.FILTER == 'topK':
            end = qa_config.ir_config.FILTER_VAR + interval
        else:
            end = 200 + interval
        tune_range = range(interval, end, interval)

    qa_tune_dp = join(path_parser.summary_rank, qa_config.QA_TUNE_DIR_NAME_BERT)
    qa_tune_result_fp = join(path_parser.tune, qa_config.QA_TUNE_DIR_NAME_BERT)
    with open(qa_tune_result_fp, mode='a', encoding='utf-8') as out_f:
        headline = 'Filter\tRecall\tF1\n'
        out_f.write(headline)

    for filter_var in tune_range:
        if exists(qa_tune_dp):  # remove previous output
            shutil.rmtree(qa_tune_dp)
        os.mkdir(qa_tune_dp)

        for cid in tqdm(cids):
            retrieval_params = {
                'model_name': qa_config.QA_MODEL_NAME_BERT,
                'cid': cid,
                'filter_var': filter_var,
                'filter': qa_config.FILTER,
                'deduplicate': None,
            }

            retrieved_items = ir_tools.retrieve(**retrieval_params)
            summary = '\n'.join([item[-1] for item in retrieved_items])
            # print(summary)
            with open(join(qa_tune_dp, cid), mode='a', encoding='utf-8') as out_f:
                out_f.write(summary)

        performance = rouge.compute_rouge_for_dev(qa_tune_dp, tune_centrality=False)
        with open(qa_tune_result_fp, mode='a', encoding='utf-8') as out_f:
            if qa_config.FILTER == 'conf':
                rec = '{0:.2f}\t{1}\n'.format(filter_var, performance)
            else:
                rec = '{0}\t{1}\n'.format(filter_var, performance)

            out_f.write(rec)


def select_e2e():
    params = {
        'model_name': qa_config.QA_MODEL_NAME_BERT,
        'cos_threshold': 1.0,
    }
    select_sent.select_for_ablation_study(**params)


def select_e2e_tdqfs():
    params = {
        'model_name': qa_config.QA_MODEL_NAME_BERT,
        'cos_threshold': 1.0,
        'cc_ids': cids,
        'ref_dp': path_parser.data_tdqfs_summary_targets,
    }
    select_sent.select_for_ablation_study(**params)


if __name__ == '__main__':
    # init()
    # dump_pid2rel_scores()
    # rel_scores2rank()

    # tune()
    # qa_rank2records()  # with selected hyper-\parameter
    qa_rank2records_in_batch()
    # select_e2e()

    select_e2e_tdqfs()
