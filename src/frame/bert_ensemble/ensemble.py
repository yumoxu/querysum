import sys
from os.path import join, dirname, abspath, exists

sys_path = dirname(dirname(abspath(__file__)))
parent_sys_path = dirname(sys_path)

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

import io
from tqdm import tqdm
import math
import numpy as np
import os

import utils.config_loader as config
from utils.config_loader import logger, path_parser
import utils.tools as tools
import tools.general_tools as general_tools
import frame.ir.ir_tools as ir_tools
import frame.bert_ensemble.ensemble_config as ensemble_config
import summ.rank_sent as rank_sent

use_tdqfs = 'tdqfs' == config.test_year
if use_tdqfs:
    query_fp = path_parser.data_tdqfs_queries
    test_cid_query_dicts = general_tools.build_tdqfs_cid_query_dicts(query_fp=query_fp, proc=True)
    cids = [cq_dict['cid'] for cq_dict in test_cid_query_dicts]
else:
    cids = tools.get_test_cc_ids()


def _read_records(cid):
    def _get_sent2score(qa_record_fp, proc_passage):
        qa_record = io.open(qa_record_fp).readlines()
        sent2score = {}
        for line in qa_record:
            _, score, sent = line.strip('\n').split('\t')
            score = float(score)

            if proc_passage:
                score = math.tanh(math.sqrt(score))

            if not 0 <= score <= 1:
                raise ValueError('Invalid score: {}'.format(score))

            if sent in sent2score and score < sent2score[sent]:
                continue
            sent2score[sent] = score
        return sent2score

    sent2score_s = _get_sent2score(qa_record_fp=join(ensemble_config.SENT_QA_RECORD_DP, cid), proc_passage=False)
    sent2score_p = _get_sent2score(qa_record_fp=join(ensemble_config.PASSAGE_QA_RECORD_DP, cid), proc_passage=True)

    return sent2score_s, sent2score_p


def _proc_one_side_score(score):
    if ensemble_config.IS_SENT_REC_ONLY:  # then this score is from sent records
        if ensemble_config.ENSEMBLE_MODE == 'weight_avg_sent_only':
            return (1 - ensemble_config.SPAN_REC_WEIGHT) * score
        else:
            raise ValueError('Corrupted conditions!')

    if not ensemble_config.IS_ENSEMBLE_GLOBAL:
        return score
    elif ensemble_config.ENSEMBLE_MODE == 'sqrt_global':
        return 0.001 * score
    elif ensemble_config.ENSEMBLE_MODE == 'avg_global':
        return score / 2
    else:
        raise ValueError('Corrupted conditions!')


def _proc_two_side_scores(score_s, score_p):
    if ensemble_config.ENSEMBLE_MODE in ('sqrt', 'sqrt_global'):
        return math.sqrt(score_s * score_p)
    elif ensemble_config.ENSEMBLE_MODE in ('avg', 'avg_global'):
        return (score_s + score_p) / 2
    elif ensemble_config.ENSEMBLE_MODE == 'weight_avg_sent_only':
        return (1 - ensemble_config.SPAN_REC_WEIGHT) * score_s + ensemble_config.SPAN_REC_WEIGHT * score_p
    else:
        raise ValueError('Invalid ENSEMBLE_MODE: {}'.format(ensemble_config.ENSEMBLE_MODE))


def _ensemble_records(cid):
    sent2score_s, sent2score_p = _read_records(cid)

    # n_sents_p = len(sent2score_p)
    sent2score_ensemble = {}
    for sent, score_s in sent2score_s.items():
        if sent not in sent2score_p:
            score_s = _proc_one_side_score(score_s)
            if score_s:
                sent2score_ensemble[sent] = score_s
            continue

        score_p = sent2score_p[sent]
        sent2score_ensemble[sent] = _proc_two_side_scores(score_s, score_p)
        del sent2score_p[sent]

    # logger.info('n_sents_p: {} -> {}'.format(n_sents_p, len(sent2score_p)))
    if not ensemble_config.IS_SENT_REC_ONLY:
        for sent, score_p in sent2score_p.items():
            score_p = _proc_one_side_score(score_p)
            if score_p:
                sent2score_ensemble[sent] = score_p

    return sent2score_ensemble


def _rank(cid):
    sent2score_ensemble = _ensemble_records(cid)
    sent_score_list = sorted(sent2score_ensemble.items(), key=lambda item: item[1], reverse=True)

    records = []
    for sid, sent_score in enumerate(sent_score_list):
        rec = ('0_{}'.format(sid), str(sent_score[1]), sent_score[0])
        records.append('\t'.join(rec))

    return records


def rank():
    rank_dp = join(path_parser.summary_rank, ensemble_config.MODEL_NAME)
    if exists(rank_dp):
        raise ValueError('rank_dp exists: {}'.format(rank_dp))
    os.mkdir(rank_dp)

    for cid in tqdm(cids):
        rank_records = _rank(cid)
        n_sents = rank_sent.dump_rank_records(rank_records=rank_records, out_fp=join(rank_dp, cid), with_rank_idx=False)
        logger.info('Dump {} ranking records'.format(n_sents))


def rank2records():
    rec_dp = join(path_parser.summary_rank, ensemble_config.QA_RECORD_DIR_NAME)

    if exists(rec_dp):
        raise ValueError('rec_dp exists: {}'.format(rec_dp))
    os.mkdir(rec_dp)

    for cid in tqdm(cids):
        retrieval_params = {
            'model_name': ensemble_config.MODEL_NAME,
            'cid': cid,
            'filter_var': ensemble_config.FILTER_VAR,
            'filter': ensemble_config.FILTER,
            'deduplicate': None,
        }

        retrieved_items = ir_tools.retrieve(**retrieval_params)
        ir_tools.dump_retrieval(fp=join(rec_dp, cid), retrieved_items=retrieved_items)


def rank2records_in_batch():
    interval = 10
    start = 20
    end = 150 + interval
    filter_var_range = range(start, end, interval)

    for filter_var in tqdm(filter_var_range):
        qa_rec_dn = ensemble_config.QA_RECORD_DIR_NAME_PATTERN.format(ensemble_config.MODEL_NAME,
                                                                      filter_var,
                                                                      ensemble_config.FILTER)
        qa_rec_dp = join(path_parser.summary_rank, qa_rec_dn)

        if exists(qa_rec_dp):
            raise ValueError('qa_rec_dp exists: {}'.format(qa_rec_dp))
        os.mkdir(qa_rec_dp)

        for cid in cids:
            retrieval_params = {
                'model_name': ensemble_config.MODEL_NAME,
                'cid': cid,
                'filter_var': filter_var,
                'filter': ensemble_config.FILTER,
                'deduplicate': None,
            }

            retrieved_items = ir_tools.retrieve(**retrieval_params)
            ir_tools.dump_retrieval(fp=join(qa_rec_dp, cid), retrieved_items=retrieved_items)


if __name__ == '__main__':
    rank()
    rank2records()
    # rank2records_in_batch()
