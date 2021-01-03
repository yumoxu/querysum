import utils.config_loader as config
from utils.config_loader import config_model
from data.dataset_parser import dataset_parser
import numpy as np


def _build_bert_tokens_for_sent(query_tokens, instance_tokens):
    in_size = [config_model['max_n_tokens'], ]

    token_ids = np.zeros(in_size, dtype=np.int32)
    seg_ids = np.zeros(in_size, dtype=np.int32)
    token_masks = np.zeros(in_size)

    # logger.info('shape of token_ids: {}'.format(token_ids.shape))
    tokens = ['[CLS]'] + query_tokens + ['[SEP]'] + instance_tokens + ['[SEP]']
    # logger.info('tokens: {}'.format(tokens))
    token_id_list = config.bert_tokenizer.convert_tokens_to_ids(tokens)
    n_tokens = len(token_id_list)

    token_ids[:n_tokens] = token_id_list
    seg_ids[len(query_tokens) + 2:n_tokens] = [1] * (len(instance_tokens) + 1)
    token_masks[:n_tokens] = [1] * n_tokens

    sent_in = {
        'token_ids': token_ids,
        'seg_ids': seg_ids,
        'token_masks': token_masks,
    }

    return sent_in


def build_instance_tokens_with_context(sent_idx, doc_sents, window):
    if window <= 0:
        raise ValueError('Invalid window: {}'.format(window))

    n_sent = len(doc_sents)
    context = []

    context_idx = 0
    context_token_pat = '[unused{}] '

    for i in range(window):
        # preceding
        idx_a = sent_idx - i - 1
        context_idx += 1
        context_token = context_token_pat.format(context_idx)

        if idx_a >= 0:
            context.append(context_token + doc_sents[idx_a])
        else:
            context.append(context_token)

        # subsequent
        idx_b = sent_idx + i + 1
        context_idx += 1
        context_token = context_token_pat.format(context_idx)
        if idx_b < n_sent:
            context.append(context_token + doc_sents[idx_b])
        else:
            context.append(context_token)

    sent = doc_sents[sent_idx]
    context.insert(0, sent)
    sent = ' '.join(context)

    return sent


def build_bert_x(query, doc_fp, window=None):
    # prep resources: query and document
    query_tokens = dataset_parser.parse_query(query)

    doc_res = dataset_parser.parse_doc2sents(doc_fp)
    in_size = [config_model['max_ns_doc'], config_model['max_n_tokens']]
    token_ids = np.zeros(in_size, dtype=np.int32)
    seg_ids = np.zeros(in_size, dtype=np.int32)
    token_masks = np.zeros(in_size, dtype=np.float32)

    # concat sentence with query
    for sent_idx in range(doc_res['sents']):
        instance_tokens = build_instance_tokens_with_context(sent_idx,
                                                         doc_sents=doc_res['sents'],
                                                         window=window)
        sent_in = _build_bert_tokens_for_sent(query_tokens=query_tokens,
                                              instance_tokens=instance_tokens)
        token_ids[sent_idx] = sent_in['token_ids']
        seg_ids[sent_idx] = sent_in['seg_ids']
        token_masks[sent_idx] = sent_in['token_masks']

    xx = {
        'token_ids': token_ids,
        'seg_ids': seg_ids,
        'token_masks': token_masks,
        'doc_masks': doc_res['doc_masks'],
    }

    return xx


def build_bert_sentence_x(query, sentence):
    query_tokens = dataset_parser.parse_query(query)
    instance_tokens = dataset_parser.sent2words(sentence)[:config_model['max_nw_sent']]
    return _build_bert_tokens_for_sent(query_tokens, instance_tokens)
