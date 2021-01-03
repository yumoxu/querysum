# -*- coding: utf-8 -*-
from os.path import dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import utils.config_loader as config
import utils.config_loader as config
import utils.tools as tools

import torch
import shutil


versions = ['sl', 'alpha']
para_org = True
for vv in versions:
    if config.meta_model_name.endswith(vv):
        para_org = False


def sort_sid2score(sid2score):
    sid_score_list = sorted(sid2score.items(), key=lambda item: item[1], reverse=True)
    return sid_score_list


def get_rank_records(sid_score_list, sents=None, flat_sents=False):
    """
        optional: display sentence in record
    :param sid_score_list:
    :param sents:
    :param flat_sents: if True, iterate sent directly; if False, need use sid to get doc_idx and sent_idx.
    :return:
    """
    rank_records = []
    for sid, score in sid_score_list:
        items = [sid, str(score)]
        if sents:
            if flat_sents:
                sent = sents[len(rank_records)]  # the current point
            else:
                doc_idx, sent_idx = tools.get_sent_info(sid)
                sent = sents[doc_idx][sent_idx]
            items.append(sent)
        record = '\t'.join(items)
        rank_records.append(record)
    return rank_records


def dump_rank_records(rank_records, out_fp, with_rank_idx):
    """
        each line is
            ranking  sid   score

        sid: config.SEP.join((doc_idx, para_idx, sent_idx))
    :param sid_score_list:
    :param out_fp:
    :return:
    """
    lines = rank_records
    if with_rank_idx:
        lines = ['\t'.join((str(rank), record)) for rank, record in enumerate(rank_records)]

    with open(out_fp, mode='a', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return len(lines)
