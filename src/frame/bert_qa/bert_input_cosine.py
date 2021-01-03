import utils.config_loader as config
from utils.config_loader import config_model
from data.dataset_parser import dataset_parser
import numpy as np


def _build_bert_in(tokens):
    in_size = [config_model['max_n_tokens'], ]

    token_ids = np.zeros(in_size, dtype=np.int32)
    seg_ids = np.zeros(in_size, dtype=np.int32)
    token_masks = np.zeros(in_size)

    tokens = ['[CLS]'] + tokens + ['[SEP]']
    token_id_list = config.bert_tokenizer.convert_tokens_to_ids(tokens)
    n_tokens = len(token_id_list)

    token_ids[:n_tokens] = token_id_list
    token_masks[:n_tokens] = [1] * n_tokens

    bert_in = {
        'token_ids': token_ids,
        'seg_ids': seg_ids,
        'token_masks': token_masks,
    }

    return bert_in


def build_query(query):
    query_tokens = dataset_parser.parse_query(query)
    return _build_bert_in(query_tokens)


def build_sentence(sentence):
    sentence_tokens = dataset_parser.sent2words(sentence)[:config_model['max_nw_sent']]
    return _build_bert_in(sentence_tokens)
