# -*- coding: utf-8 -*-
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from os.path import join, isdir, dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import utils.config_loader as config
from utils.config_loader import path_parser, logger, config_model, config_meta

from data.dataset_parser import dataset_parser
import data.bert_input as bert_in
import data.bert_input_sep as bert_input_sep
import data.bert_input_sl as bert_input_sl
import data.data_tools as data_tools
import utils.tools as tools


class To2DMat(object):
    def __call__(self, numpy_dict):
        for (k, v) in numpy_dict.items():
            # logger.info('[BEFORE TO TENSOR] type of {0}: {1}'.format(k, v.dtype))
            if k in ('token_ids', 'seg_ids', 'token_masks'):
                numpy_dict[k] = v.reshape(-1, config_model['max_n_tokens'])

        return numpy_dict

