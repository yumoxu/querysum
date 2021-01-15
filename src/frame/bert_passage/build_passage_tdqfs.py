# -*- coding: utf-8 -*-
import sys
import io
import os
from os import listdir
from os.path import join, dirname, abspath, exists

sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

import tools.general_tools as general_tools
import utils.config_loader as config
import utils.tools as tools
from utils.config_loader import logger, path_parser, config_model

from data.dataset_parser import dataset_parser
from tqdm import tqdm
import dill
import itertools
from frame.bert_passage.passage_obj import SentObj, PassageObj


sentence_dp = path_parser.data_tdqfs_sentences
passages_dp = path_parser.data_tdqfs_passages
query_fp = path_parser.data_tdqfs_queries
test_cid_query_dicts = general_tools.build_tdqfs_cid_query_dicts(query_fp=query_fp, proc=True)


def get_sentences(cid):
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


def _passage_core(cid, query, passage_size, stride):
    original_sents, processed_sents = get_sentences(cid)  # 2d lists, docs => sents
    # logger.info('#doc: {}'.format(len(original_sents)))

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
            po = PassageObj(pid=pid, query=query, narr=query, sent_objs=target_sent_objs)
            passage_objs.append(po)

            start += stride

    return passage_objs


def _dump_passages(cid, passage_objs):
    cc_dp = join(passages_dp, cid)
    if not exists(cc_dp):  # remove previous output
        os.mkdir(cc_dp)

    for po in passage_objs:
        with open(join(cc_dp, po.pid), 'wb') as f:
            dill.dump(po, f)

    logger.info('[_dump_passages] Dump {} passage objects to {}'.format(len(passage_objs), cc_dp))


def build_passages(passage_size, stride):
    """

    :param passage_size: max number of sentences in a passage
    :param use_stride:
    :return:
    """
    for cid_query_dict in tqdm(test_cid_query_dicts):
        core_params = {
            **cid_query_dict,
            'passage_size': passage_size,
            'stride': stride,
        }
        passage_objs = _passage_core(**core_params)
        _dump_passages(cid=cid_query_dict['cid'], passage_objs=passage_objs)


if __name__ == '__main__':
    build_passages(passage_size=config_model['ns_passage'], 
        stride=config_model['stride'])
