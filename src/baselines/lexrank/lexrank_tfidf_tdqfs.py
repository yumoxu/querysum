# -*- coding: utf-8 -*-
import sys
import io
import os
from os import listdir
from os.path import join, dirname, abspath, exists

sys_path = dirname(dirname(abspath(__file__)))
parent_sys_path = dirname(sys_path)

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

import utils.config_loader as config
from utils.config_loader import logger, path_parser
from data.dataset_parser import dataset_parser
import utils.tools as tools
import summ.rank_sent as rank_sent
import summ.select_sent as select_sent

from lexrank import STOPWORDS, LexRank
import itertools
from tqdm import tqdm

import tools.general_tools as general_tools
from utils.tools import get_text_dp_for_tdqfs
import summ.compute_rouge as rouge

assert config.grain == 'sent'
MODEL_NAME = 'lexrank-{}'.format(config.test_year)
COS_THRESHOLD = 1.0

assert 'tdqfs' in config.test_year
sentence_dp = path_parser.data_tdqfs_sentences
query_fp = path_parser.data_tdqfs_queries
tdqfs_summary_target_dp = path_parser.data_tdqfs_summary_targets

test_cid_query_dicts = general_tools.build_tdqfs_cid_query_dicts(query_fp=query_fp, proc=True)
cc_ids = [cq_dict['cid'] for cq_dict in test_cid_query_dicts]
LENGTH_BUDGET_TUPLE = ('nw', 250)

def _get_sentences(cid):
    cc_dp = join(sentence_dp, cid)
    fns = [fn for fn in listdir(cc_dp)]
    lines = itertools.chain(*[io.open(join(cc_dp, fn)).readlines() for fn in fns])
    sentences = [line.strip('\n') for line in lines]

    original_sents = []
    processed_sents = []
    for ss in sentences:
        ss_origin = dataset_parser._proc_sent(ss, rm_dialog=False, rm_stop=False, stem=False)
        ss_proc = dataset_parser._proc_sent(ss, rm_dialog=False, rm_stop=True, stem=True)

        if ss_proc:  # make sure the sent is not removed, i.e., is not empty and is not in a dialog
            original_sents.append(ss_origin)
            processed_sents.append(ss_proc)
    
    return [original_sents], [processed_sents]


def _lexrank(cid):
    """
        Run LexRank on all sentences from all documents in a cluster.

    :param cid:
    :return: rank_records
    """
    _, processed_sents = dataset_parser.cid2sents_tdqfs(cid)  # 2d lists, docs => sents
    flat_processed_sents = list(itertools.chain(*processed_sents))  # 1d sent list

    lxr = LexRank(processed_sents, stopwords=STOPWORDS['en'])
    scores = lxr.rank_sentences(flat_processed_sents, threshold=None, fast_power_method=True)

    sid2score = dict()
    abs_idx = 0
    for doc_idx, doc in enumerate(processed_sents):
        for sent_idx, sent in enumerate(doc):
            sid = config.SEP.join((str(doc_idx), str(sent_idx)))
            score = scores[abs_idx]
            sid2score[sid] = score

            abs_idx += 1

    sid_score_list = rank_sent.sort_sid2score(sid2score)
    rank_records = rank_sent.get_rank_records(sid_score_list, sents=processed_sents, flat_sents=False)
    return rank_records


def rank_e2e():
    rank_dp = tools.get_rank_dp(model_name=MODEL_NAME)
    if exists(rank_dp):
        raise ValueError('rank_dp exists: {}'.format(rank_dp))
    os.mkdir(rank_dp)

    for cid in tqdm(cc_ids):
        rank_records = _lexrank(cid)
        rank_sent.dump_rank_records(rank_records, out_fp=join(rank_dp, cid), with_rank_idx = False)

    logger.info('[RANK SENT] successfully dumped rankings to: {}'.format(rank_dp))


def select_e2e():
    """
        No redundancy removal is applied here.
    """
    params = {
        'model_name': MODEL_NAME,
        'cos_threshold': COS_THRESHOLD,
    }
    select_sent.select_end2end(**params)


def select_e2e_tdqfs():
    params = {
        'model_name': MODEL_NAME,
        'length_budget_tuple': LENGTH_BUDGET_TUPLE,
        'cos_threshold': COS_THRESHOLD,
        'cc_ids': cc_ids,
    }
    select_sent.select_end2end_for_tdqfs(**params)


def compute_rouge_tdqfs(length):
    text_params = {
        'model_name': MODEL_NAME,
        'length_budget_tuple': LENGTH_BUDGET_TUPLE,
        'cos_threshold': COS_THRESHOLD,
    }

    text_dp = get_text_dp_for_tdqfs(**text_params)

    rouge_parmas = {
        'text_dp': text_dp,
        'ref_dp': tdqfs_summary_target_dp,
    }
    if LENGTH_BUDGET_TUPLE[0] == 'nw':
        rouge_parmas['length'] = LENGTH_BUDGET_TUPLE[1]
    
    output = rouge.compute_rouge_for_tdqfs(**rouge_parmas)
    return output


if __name__ == '__main__':
    rank_e2e()
    select_e2e_tdqfs()
    compute_rouge_tdqfs(length=None)
