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
from utils.config_loader import path_parser
import dill

class SentObj:
    def __init__(self, sid, original_sent, proc_sent):
        self.sid = sid  # config.SEP.join([cid, str(sent_idx)]); for score gathering in QA module
        self.original_sent = original_sent
        self.proc_sent = proc_sent


class PassageObj:
    def __init__(self, pid, query, narr, sent_objs):
        self.pid = pid  # config.SEP.join([cid, str(passage_idx)])
        self.query = query
        self.narr = narr

        self.sent_objs = sent_objs
        self.size = len(self.sent_objs)

        self.ir_score = None

    def get_original_sents(self):
        return [so.original_sent for so in self.sent_objs]

    def get_proc_sents(self):
        return [so.proc_sent for so in self.sent_objs]

    def get_proc_passage(self):
        return ' '.join(self.get_proc_sents())

    def get_original_passage(self):
        return ' '.join(self.get_original_sents())


def pid2obj(cid, pid, use_tdqfs):
    if use_tdqfs:
        fp = join(path_parser.data_tdqfs_passages, cid, pid)
    else:
        year, _ = cid.split(config.SEP)
        fp = join(path_parser.data_passages, year, cid, pid)
    
    with open(fp, 'rb') as f:
        po = dill.load(f)
    return po
