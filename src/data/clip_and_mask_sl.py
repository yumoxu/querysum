from os.path import dirname, abspath
import sys
from utils.config_loader import config_model
import numpy as np

sys.path.insert(0, dirname(dirname(abspath(__file__))))

"""
    for doc: clip_doc_sents, mask_doc_sents, clip_and_mask_doc_sents
"""


def _len2mask(lens, mask_shape, offset):
    """
        could be applied to:
            [1] paragraph masks: pooling paragraph instance scores to document bag score
            [2] sentence masks: pooling word representations to sentence representation
    :param lens: n_sents or n_paras
    :param mask_shape: [max_n_sents, max_words] or [max_n_docs, max_n_paras]
    :return:
    """
    mask = np.zeros(mask_shape, dtype=np.float32)
    if type(lens) != list:
        raise ValueError('Invalid lens type: {}'.format(type(lens)))

    if len(mask_shape) == 1:  # mask a document with its paras
        mask[offset:offset + lens[0]] = [1] * lens[0]
        return mask

    elif len(mask_shape) == 2:  # mask sentences of a para/query with their words
        for idx, ll in enumerate(lens):
            end = offset + ll
            mask[idx, offset:end] = [1] * ll
            offset = end
        return mask

    else:
        raise ValueError('Invalid mask dim: {}'.format(len(mask_shape)))


def mask_para(n_sents, max_n_sents):
    """

    :param n_sents: an int.
    :param max_n_sents:
    :return:
    """
    mask_shape = [max_n_sents, ]
    return _len2mask([n_sents], mask_shape=mask_shape, offset=0)


def clip_doc_sents(sents):
    """
        For QueryNetSL
    :param sents:
    :return:
    """

    words = [ss[:config_model['max_nw_sent']] for ss in sents[:config_model['max_ns_doc']]]
    n_words = [len(ss) for ss in words]

    res = {
        'words': words,
        'n_words_by_sents': n_words,
    }
    return res


def mask_doc_sents(n_words):
    """
        mask sentences of a doc with their words.

    :param n_words: an int list of sentence sizes in words.
    """
    mask_shape = (config_model['max_ns_doc'], config_model['max_nw_sent'])
    # logger.info('mask shape: {}'.format(mask_shape))
    return _len2mask(n_words, mask_shape=mask_shape, offset=0)


def clip_and_mask_doc_sents(sents):
    """
        For QueryNetSL.

    :param sents:
    :param offset:
    :return:
    """
    clipped_res = clip_doc_sents(sents)
    doc_masks = mask_para(len(clipped_res['n_words_by_sents']),
                          max_n_sents=config_model['max_ns_doc'])

    res = {
        'sents': clipped_res['words'],
        'doc_masks': doc_masks,  # max_ns_doc,
    }
    return res
