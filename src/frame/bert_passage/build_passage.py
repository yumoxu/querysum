# -*- coding: utf-8 -*-
import sys

import os
from os.path import join, dirname, abspath, exists

sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

import utils.config_loader as config
import utils.tools as tools
from utils.config_loader import logger, path_parser, config_model
from data.dataset_parser import dataset_parser
from tqdm import tqdm
import dill
from frame.bert_passage.passage_obj import SentObj, PassageObj


def _passage_core(cid, query, narr, passage_size, stride):
    original_sents, processed_sents = dataset_parser.cid2sents(cid, max_ns_doc=None)  # 2d lists, docs => sents
    logger.info('#doc: {}'.format(len(original_sents)))

    # build sent_objs
    sent_objs = []  # organized by doc
    sent_idx = 0
    for doc_idx in range(len(original_sents)):
        sent_objs_doc = []
        for original_s, proc_s in zip(original_sents[doc_idx], processed_sents[doc_idx]):
            sid = config.SEP.join([cid, str(sent_idx)])
            so = SentObj(sid=sid, original_sent=original_s, proc_sent=proc_s)
            sent_objs_doc.append(so)
            sent_idx += 1

        sent_objs.append(sent_objs_doc)

    # build passage objs
    passage_objs = []
    for sent_objs_doc in sent_objs:
        start = 0
        # make sure the last sentence whose length < stride will be discarded
        while start + stride < len(sent_objs_doc):
            pid = config.SEP.join([cid, str(len(passage_objs))])

            target_sent_objs = sent_objs_doc[start:start+passage_size]
            po = PassageObj(pid=pid, query=query, narr=narr, sent_objs=target_sent_objs)
            passage_objs.append(po)

            start += stride

    return passage_objs


def _dump_passages(year, cid, passage_objs):
    cc_dp = join(path_parser.data_passages, year, cid)
    if not exists(cc_dp):  # remove previous output
        os.mkdir(cc_dp)

    for po in passage_objs:
        with open(join(cc_dp, po.pid), 'wb') as f:
            dill.dump(po, f)

    logger.info('[_dump_passages] Dump {} passage objects to {}'.format(len(passage_objs), cc_dp))


def build_passages(passage_size, stride, year=None):
    """

    :param passage_size: max number of sentences in a passage
    :param use_stride:
    :return:
    """
    query_info = dataset_parser.get_cid2query(tokenize_narr=False)
    narr_info = dataset_parser.get_cid2narr()

    if year:
        years = [year]
    else:
        years = config.years

    for year in years:
        cc_ids = tools.get_cc_ids(year, model_mode='test')

        for cid in tqdm(cc_ids):
            core_params = {
                'cid': cid,
                'query': query_info[cid],
                'narr': narr_info[cid],
                'passage_size': passage_size,
                'stride': stride,
            }
            passage_objs = _passage_core(**core_params)

            _dump_passages(year, cid, passage_objs)


if __name__ == '__main__':
    build_passages(passage_size=config_model['ns_passage'], stride=config_model['stride'], year='2007')
