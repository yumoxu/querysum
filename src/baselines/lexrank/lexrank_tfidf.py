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

import utils.config_loader as config
from utils.config_loader import logger
from data.dataset_parser import dataset_parser
import utils.tools as tools
import summ.rank_sent as rank_sent
import summ.select_sent as select_sent

from lexrank import STOPWORDS, LexRank
import itertools
from tqdm import tqdm

assert config.grain == 'sent'
MODEL_NAME = 'lexrank-{}'.format(config.test_year)
COS_THRESHOLD = 1.0

def _lexrank(cid):
    """
        Run LexRank on all sentences from all documents in a cluster.

    :param cid:
    :return: rank_records
    """
    _, processed_sents = dataset_parser.cid2sents(cid)  # 2d lists, docs => sents
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

    cc_ids = tools.get_test_cc_ids()
    for cid in tqdm(cc_ids):
        rank_records = _lexrank(cid)
        rank_sent.dump_rank_records(rank_records, out_fp=join(rank_dp, cid), with_rank_idx = False)

    logger.info('Successfully dumped rankings to: {}'.format(rank_dp))


def select_e2e():
    """
        No redundancy removal is applied here.
    """
    params = {
        'model_name': MODEL_NAME,
        'cos_threshold': COS_THRESHOLD,
    }
    select_sent.select_end2end(**params)

if __name__ == '__main__':
    rank_e2e()
    select_e2e()
