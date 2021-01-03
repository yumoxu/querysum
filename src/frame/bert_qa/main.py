# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname, abspath, exists

sys_path = dirname(dirname(abspath(__file__)))
parent_sys_path = dirname(sys_path)

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

import utils.config_loader as config
from utils.config_loader import logger, path_parser, config_meta
from os.path import join, dirname, abspath, exists
import frame.bert_qa.data_pipe_cluster as data_pipe_cluster
import frame.bert_qa.data_pipe_cluster_cosine as data_pipe_cluster_cosine

import copy
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils.tools as tools

from argparse import ArgumentParser
import os
import io
from tqdm import tqdm
import summ.rank_sent as rank_sent
import summ.select_sent as select_sent
import frame.ir.ir_tools as ir_tools
from frame.ir.ir_tools import load_retrieved_sentences
import frame.bert_qa.qa_config as qa_config
import frame.centrality.centrality_config as centrality_config
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import summ.compute_rouge as rouge

import tools.general_tools as general_tools

"""
    This module builds the following pipeline:

    rel_scores [compute, dump] =>
    qa_rank [load rel_scores, compute, dump]=>
    qa_records [load qa_rank, compute, dump]

    For the same IR records and query type, rel_scores and qa_rank only need to be processed once.

    qa_records can be generated with different CONF_THRESHOLD_QA.

"""
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

# CPU_INFERENCE = True  # temperal use
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


def _dump(model, cluster_loader, dump_dp):
    doc_rel_scores = []
    for _, batch in enumerate(cluster_loader):
        feed_dict = copy.deepcopy(batch)

        for (k, v) in feed_dict.items():
            with torch.no_grad():
                feed_dict[k] = Variable(v, requires_grad=False)

        n_sents, max_nt = feed_dict['token_ids'].size()
        # pred: (batch * max_ns_doc) * 2
        pred = model(feed_dict['token_ids'],
                     feed_dict['seg_ids'],
                     feed_dict['token_masks'])

        if type(pred) is tuple:  # BertForSequenceClassification returns tuple
            pred = pred[0]

        n_cls = pred.size()[-1]
        if n_cls == 2:
            pred = F.softmax(pred, dim=-1)[:, 1]
        elif n_cls == 1:
            pred = pred.squeeze(-1)
        else:
            raise ValueError('Invalid n_cls: {}'.format(n_cls))

        rel_scores = pred.cpu().detach().numpy()  # d_batch,
        # logger.info('rel_scores: {}'.format(rel_scores.shape))

        doc_rel_scores.append(rel_scores[:n_sents])
    
    rel_scores = np.concatenate(doc_rel_scores)
    dump_fp = join(dump_dp, cluster_loader.cid)
    tools.save_obj(obj=rel_scores, fp=dump_fp)
    # logger.info('dumping ranking file to: {0}'.format(dump_fp))


def _dump_cos(model, cluster_loader, dump_dp, use_cls):
    doc_rel_scores = []
    feed_dict = copy.deepcopy(cluster_loader.query_in)
    for (k, v) in feed_dict.items():
        with torch.no_grad():
            feed_dict[k] = Variable(v, requires_grad=False)

    pred = model(feed_dict['token_ids'],
                 feed_dict['seg_ids'],
                 feed_dict['token_masks'])

    if use_cls:
        query_vec = pred[1].cpu().detach().numpy()  # d_batch
    else:
        query_vec = pred[0].cpu().detach().numpy()  # d_batch
        raise ValueError('use_cls==False is not implemented')

    for _, batch in enumerate(cluster_loader):
        feed_dict = copy.deepcopy(batch)

        for (k, v) in feed_dict.items():
            with torch.no_grad():
                feed_dict[k] = Variable(v, requires_grad=False)

        n_sents, max_nt = feed_dict['token_ids'].size()
        pred = model(feed_dict['token_ids'],
                     feed_dict['seg_ids'],
                     feed_dict['token_masks'])

        if type(pred) is not tuple:  # BertForSequenceClassification returns tuple
            raise ValueError('Invalid type of pred: {}'.format(type(pred)))

        # 0: sequence_output, 1: pooled_output
        if use_cls:
            sent_vecs = pred[1].cpu().detach().numpy()  # d_batch
            sent_vecs = sent_vecs[:n_sents]
        else:
            sent_vecs = pred[0].cpu().detach().numpy()  # d_batch
            raise ValueError('use_cls==False is not implemented')

        print('query_vec: {}, sent_vecs: {}'.format(query_vec.shape, sent_vecs.shape))
        rel_scores = cosine_similarity(sent_vecs, query_vec).reshape(-1,)
        logger.info('[_dump] rel_scores: {}'.format(rel_scores.shape))
        doc_rel_scores.append(rel_scores)
    rel_scores = np.concatenate(doc_rel_scores)

    dump_fp = join(dump_dp, cluster_loader.cid)
    tools.save_obj(obj=rel_scores, fp=dump_fp)
    logger.info('[_dump] dumping ranking file to: {0}'.format(dump_fp))


def get_data_loader_gen(ir_rec_dp):
    if use_tdqfs:    
        if config.meta_model_name in ('bert_qa'):
            data_gen = data_pipe_cluster.TdqfsQSDataLoader(test_cid_query_dicts=test_cid_query_dicts, retrieve_dp=ir_rec_dp)
            return data_gen
        elif config.meta_model_name == 'bert_base':
            data_gen = data_pipe_cluster_cosine.TdqfsQSDataLoader(test_cid_query_dicts=test_cid_query_dicts, retrieve_dp=ir_rec_dp)
            return data_gen
        else:
            raise NotImplementedError('Passage pipeline to be implemented.')
            
    loader_params = {
        'tokenize_narr': False,
        'query_type': qa_config.QUERY_TYPE,
        'retrieve_dp': ir_rec_dp,
    }

    if config.meta_model_name == 'bert_qa':  # w/o window config
        loader_cls = data_pipe_cluster.QSDataLoader
    elif config.meta_model_name == 'bert_base':
        loader_cls = data_pipe_cluster_cosine.QSDataLoader
    else:
        raise ValueError('Invalid meta_model_name: {}'.format(config.meta_model_name))

    data_gen = loader_cls(**loader_params)
    return data_gen


def dump_rel_scores():
    assert not exists(rel_scores_dp), f'rel_scores_dp exists: {rel_scores_dp}'
    os.mkdir(rel_scores_dp)

    ir_rec_dp = join(path_parser.summary_rank, qa_config.IR_RECORDS_DIR_NAME)

    model = _place_model(model=config.bert_model)
    data_loader_generator = get_data_loader_gen(ir_rec_dp)

    total = len(cids)
    for cluster_loader in tqdm(data_loader_generator, total=total):
        if config.meta_model_name == 'bert_qa':
            _dump(model, cluster_loader=cluster_loader, dump_dp=rel_scores_dp)
        elif config.meta_model_name == 'bert_base':
            _dump_cos(model, cluster_loader=cluster_loader, dump_dp=rel_scores_dp, use_cls=True)
        else:
            raise ValueError('Invalid meta_model_name: {}'.format(config.meta_model_name))


def load_rel_scores(cid, rel_scores_dp):
    rel_scores_fp = join(rel_scores_dp, cid)
    return tools.load_obj(rel_scores_fp)


def rel_scores2rank():
    if exists(rank_dp):
        raise ValueError('rank_dp exists: {}'.format(rank_dp))
    os.mkdir(rank_dp)

    for cid in tqdm(cids):
        rel_scores = load_rel_scores(cid=cid, rel_scores_dp=rel_scores_dp)
        sent_ids = np.argsort(rel_scores)[::-1].tolist()

        sid_score_list = []
        for sid in sent_ids:
            sid_score = ('0_{}'.format(sid), rel_scores[sid])
            sid_score_list.append(sid_score)

        original_sents, _ = load_retrieved_sentences(retrieved_dp=ir_rec_dp, cid=cid)
        rank_records = rank_sent.get_rank_records(sid_score_list, sents=original_sents)

        n_sents = rank_sent.dump_rank_records(rank_records=rank_records, out_fp=join(rank_dp, cid), with_rank_idx=False)
        logger.info('Dump {} ranking records'.format(n_sents))


def tune():
    """
        Tune QA confidence / topK
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


def select_e2e():
    ir_rec_dp = join(path_parser.summary_rank, qa_config.IR_RECORDS_DIR_NAME)
    params = {
        'model_name': qa_config.QA_MODEL_NAME_BERT,
        'cos_threshold': 1.0,
        'retrieved_dp': ir_rec_dp,
    }
    select_sent.select_end2end(**params)


def select_e2e_tdqfs():
    params = {
        'model_name': qa_config.QA_MODEL_NAME_BERT,
        'length_budget_tuple': ('nw', 250),
        'cos_threshold': 0.6,  # do not pos cosine similarity criterion?
        'retrieved_dp': ir_rec_dp,
        'cc_ids': cids,
    }
    select_sent.select_end2end_for_tdqfs(**params)


def compute_rouge_tdqfs():
    text_params = {
        'model_name': qa_config.QA_MODEL_NAME_BERT,
        'length_budget_tuple': ('nw', 250),
        'cos_threshold': 0.6,  # do not pos cosine similarity criterion?
    }
    text_dp = tools.get_text_dp_for_tdqfs(**text_params)

    rouge_parmas = {
        'text_dp': text_dp,
        'ref_dp': tdqfs_summary_target_dp,
    }
    if centrality_config.LENGTH_BUDGET_TUPLE[0] == 'nw':
        rouge_parmas['length'] = centrality_config.LENGTH_BUDGET_TUPLE[1]
    
    output = rouge.compute_rouge_for_tdqfs(**rouge_parmas)
    return output


def run(prep=False, mode='rec'):
    if prep:  # run only once
        init()
        dump_rel_scores()
        rel_scores2rank()
    
    if mode == 'rec':  # with selected hyper-parameter
        qa_rank2records()
    elif mode == 'tune':
        tune()
    elif mode == 'ablaton':  # w/o Centrality
        if use_tdqfs:
            select_e2e_tdqfs()
            compute_rouge_tdqfs()
        else:
            select_e2e()
    else:
        raise ValueError


if __name__ == '__main__':
    run(prep=True, mode='rec')
