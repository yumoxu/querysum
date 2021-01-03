# -*- coding: utf-8 -*-
import sys

import os
from os.path import join, dirname, abspath, exists

sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

import utils.config_loader as config
from utils.config_loader import logger, path_parser
import summ.rank_sent as rank_sent
import utils.tools as tools
import tools.tfidf_tools as tfidf_tools
import tools.general_tools as general_tools
import frame.ir.ir_tools as ir_tools

from tqdm import tqdm
import shutil
import frame.ir.ir_config as ir_config
import summ.compute_rouge as rouge
from frame.ir.ir_tools import load_retrieved_passages
import numpy as np

if not config.grain.startswith('passage'):
    raise ValueError('Invalid grain: {}'.format(config.grain))
assert ir_config.test_year.startswith('tdqfs'), f'set ir_config.test_year to tdqfs! now: {ir_config.test_year}'

query_fp = path_parser.data_tdqfs_queries
test_cid_query_dicts = general_tools.build_tdqfs_cid_query_dicts(query_fp=query_fp, proc=True)
cids = [cq_dict['cid'] for cq_dict in test_cid_query_dicts]


def _rank(cid, query):
    pid2score = tfidf_tools.build_rel_scores_tf_passage(cid, query, tdqfs_data=True)
    # rank scores
    sid_score_list = rank_sent.sort_sid2score(pid2score)
    # include sentences in records
    rank_records = rank_sent.get_rank_records(sid_score_list, sents=None)
    # rank_records = rank_sent.get_rank_records(sid_score_list)

    return rank_records


def rank_e2e():
    """

    :param pool_func: avg, max, or None (for integrated query).
    :return:
     """
    rank_dp = join(path_parser.summary_rank, ir_config.IR_MODEL_NAME_TF)

    if exists(rank_dp):
        raise ValueError('rank_dp exists: {}'.format(rank_dp))
    os.mkdir(rank_dp)

    for cid_query_dict in tqdm(test_cid_query_dicts):
        rank_records = _rank(**cid_query_dict)
        rank_sent.dump_rank_records(rank_records, out_fp=join(rank_dp, cid_query_dict['cid']), with_rank_idx=False)
    logger.info('Successfully dumped rankings to: {}'.format(rank_dp))


def ir_rank2records():
    ir_rec_dp = join(path_parser.summary_rank, ir_config.IR_RECORDS_DIR_NAME_TF)

    if exists(ir_rec_dp):
        raise ValueError('qa_rec_dp exists: {}'.format(ir_rec_dp))
    os.mkdir(ir_rec_dp)

    for cid in tqdm(cids):
        retrieval_params = {
            'model_name': ir_config.IR_MODEL_NAME_TF,
            'cid': cid,
            'filter_var': ir_config.FILTER_VAR,
            'filter': ir_config.FILTER,
            'deduplicate': ir_config.DEDUPLICATE,
            'prune': True,
        }

        retrieved_items = ir_tools.retrieve(**retrieval_params)
        ir_tools.dump_retrieval(fp=join(ir_rec_dp, cid), retrieved_items=retrieved_items)


def tune():
    """
        Tune IR confidence / compression rate based on Recall Rouge 2.
    :return:
    """
    if ir_config.FILTER == 'conf':
        tune_range = np.arange(0.05, 1.05, 0.05)
    else:
        interval = 10
        tune_range = range(interval, 500 + interval, interval)

    ir_tune_dp = join(path_parser.summary_rank, ir_config.IR_TUNE_DIR_NAME_TF)
    ir_tune_result_fp = join(path_parser.tune, ir_config.IR_TUNE_DIR_NAME_TF)
    with open(ir_tune_result_fp, mode='a', encoding='utf-8') as out_f:
        headline = 'Filter\tRecall\tF1\n'
        out_f.write(headline)

    cids = tools.get_test_cc_ids()
    for filter_var in tune_range:
        if exists(ir_tune_dp):  # remove previous output
            shutil.rmtree(ir_tune_dp)
        os.mkdir(ir_tune_dp)

        for cid in tqdm(cids):
            retrieval_params = {
                'model_name': ir_config.IR_MODEL_NAME_TF,
                'cid': cid,
                'filter_var': filter_var,
                'filter': ir_config.FILTER,
                'deduplicate': ir_config.DEDUPLICATE,
                'prune': True,
            }

            retrieved_items = ir_tools.retrieve(**retrieval_params)  # pid, score

            passage_ids = [item[0] for item in retrieved_items]
            original_passages, _, _ = load_retrieved_passages(cid=cid,
                                                              get_sents=True,
                                                              passage_ids=passage_ids)
            passages = ['\n'.join(sents) for sents in original_passages]
            summary = '\n'.join(passages)
            print(summary)
            # print(summary)
            with open(join(ir_tune_dp, cid), mode='a', encoding='utf-8') as out_f:
                out_f.write(summary)

        performance = rouge.compute_rouge_for_dev(ir_tune_dp, tune_centrality=False)
        with open(ir_tune_result_fp, mode='a', encoding='utf-8') as out_f:
            if ir_config.FILTER == 'conf':
                rec = '{0:.2f}\t{1}\n'.format(filter_var, performance)
            else:
                rec = '{0}\t{1}\n'.format(filter_var, performance)

            out_f.write(rec)


if __name__ == '__main__':
    rank_e2e()
    ir_rank2records()
    # tune()
