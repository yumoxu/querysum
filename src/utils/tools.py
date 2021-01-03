import io
import pickle
import os
from os import listdir
import itertools
from os.path import isfile, isdir, join, dirname, abspath, exists
import sys

import math
from collections import Counter

sys.path.insert(0, dirname(dirname(abspath(__file__))))
import utils.config_loader as config
from utils.config_loader import config_meta, path_parser


def flatten(list2d):
    return list(itertools.chain(*list2d))


def save_obj(obj, fp):
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)


def get_cc_ids(year, model_mode):
    root = join(path_parser.data_docs, year)

    all_cc_ids = [config.SEP.join((year, fn)) for fn in listdir(root) if isdir(join(root, fn))]
    assert model_mode == 'test'
    return all_cc_ids


def get_test_cc_ids():
    return get_cc_ids(config_meta['test_year'], model_mode='test')


def get_cc_dp(cid):
    year, cc = cid.split(config.SEP)
    cc_dp = join(path_parser.data_docs, year, cc)
    return cc_dp


def get_doc_ids(cid, remove_illegal):
    """

    :param cid:
    :param remove_illegal: remove empty docs with no preprocessed sents, e.g., filled with dialogs or quotes.
    :return:
    """
    cc_dp = get_cc_dp(cid)
    doc_ids = [config.SEP.join((cid, fn)) for fn in listdir(cc_dp)
               if isfile(join(cc_dp, fn)) and not join(cc_dp, fn).endswith('.swp')]

    illegal_doc_ids = ['2007_D0709B_APW19990124.0079', '2007_D0736H_APW19990311.0174']
    if remove_illegal:
        for ill in illegal_doc_ids:
            if ill in doc_ids:
                doc_ids.remove(ill)

    return doc_ids


def get_cid(did):
    return config.SEP.join(did.split(config.SEP)[:2])


def get_doc_fps(cid):
    cc_dp = get_cc_dp(cid)
    doc_fps = [join(cc_dp, fn) for fn in listdir(cc_dp)
               if isfile(join(cc_dp, fn)) and not join(cc_dp, fn).endswith('.swp')]
    return doc_fps


def get_doc_info(did):
    return did.split(config.SEP)


def get_cc_info(cid):
    return cid.split(config.SEP)


def get_doc_fp(did):
    year, cc, fn = get_doc_info(did)
    return join(path_parser.data_docs, year, cc, fn)


def get_doc_idx2id(cid):
    doc_ids = get_doc_ids(cid)
    doc_idx2id = dict()
    for doc_idx, doc_id in enumerate(doc_ids):
        doc_idx2id[doc_idx] = doc_id

    return doc_idx2id


def get_doc_fps_yearly(year):
    cc_ids = get_cc_ids(year, model_mode='test')
    doc_fps = list(itertools.chain(*[get_doc_fps(cid) for cid in cc_ids]))
    return doc_fps


def text_to_vec(sent_words):
    return Counter(sent_words)


def get_n_refs(cid):
    year, cc = cid.split(config.SEP)
    if year != '2005':
        return 4

    ref_fp = join(path_parser.data_summary_targets, year, cid)
    with io.open(ref_fp, encoding='utf-8') as ref_f:
        n_refs = len(ref_f.readlines())

    return n_refs


def get_sent_info(sid):
    doc_idx, sent_idx = [int(idx) for idx in sid.split(config.SEP)]
    return doc_idx, sent_idx


def get_sent(sents, sid):
    if type(sents[0]) != list:
        sents = [sents]

    doc_idx, sent_idx = get_sent_info(sid)

    if doc_idx >= len(sents):
        raise ValueError('Invalid doc_idx: {} for #doc: {}'.format(doc_idx, len(sents)))

    doc = sents[doc_idx]
    if sent_idx >= len(doc):
        raise ValueError('Invalid sent_idx: {} for #sents: {}'.format(sent_idx, len(doc)))

    sent = doc[sent_idx]
    return sent


def compute_sent_cosine(sent_words_1, sent_words_2):
    vec_1 = text_to_vec(sent_words_1)
    vec_2 = text_to_vec(sent_words_2)

    intersection = set(vec_1.keys()) & set(vec_2.keys())
    numerator = sum([vec_1[x] * vec_2[x] for x in intersection])

    sum_1 = sum([vec_1[x] ** 2 for x in vec_1.keys()])
    sum_2 = sum([vec_2[x] ** 2 for x in vec_2.keys()])
    denom = math.sqrt(sum_1) * math.sqrt(sum_2)

    if not denom:
        return 0.0
    else:
        return float(numerator) / denom


def add_extra(dn_items, extra):
    if extra is None:
        return dn_items

    if type(extra) is list:
        dn_items.extend([str(ex) for ex in extra])
    else:
        dn_items.append(str(extra))

    return dn_items


def get_dir_name_items(model_name, n_iter=None, diversity_param_tuple=None, extra=None):
    dn_items = [model_name]
    if n_iter:
        dn_items.append('{}_iter'.format(n_iter))

    if diversity_param_tuple:
        dn_items.append('_'.join([str(item) for item in diversity_param_tuple]))

    dn_items = add_extra(dn_items, extra=extra)
    return dn_items


def get_rank_dp(model_name, n_iter=None, diversity_param_tuple=None, extra=None):
    dn_items = get_dir_name_items(model_name, n_iter, diversity_param_tuple=diversity_param_tuple, extra=extra)
    rank_dp = join(path_parser.summary_rank, '-'.join(dn_items))
    return rank_dp


def get_text_dp(model_name,
                cos_threshold,
                diversity_param_tuple=None,
                n_iter=None,
                extra=None):
    dn_items = [model_name, '{}_cos'.format(cos_threshold)]
    if n_iter:
        dn_items.append('{}_iter'.format(n_iter))

    if diversity_param_tuple:
        dn_items.append('_'.join([str(item) for item in diversity_param_tuple]))

    dn_items = add_extra(dn_items, extra=extra)
    text_dp = join(path_parser.summary_text, '-'.join(dn_items))

    return text_dp


def get_text_dp_for_tdqfs(model_name,
        cos_threshold,
        diversity_param_tuple=None,
        length_budget_tuple=None,
        n_iter=None,
        extra=None):
    dn_items = [model_name, f'{cos_threshold}_cos']
    if n_iter:
        dn_items.append(f'{n_iter}_iter')

    if diversity_param_tuple:
        dn_items.append('_'.join([str(item) for item in diversity_param_tuple]))

    if length_budget_tuple:
        dn_items.append('_'.join([str(item) for item in length_budget_tuple]))

    dn_items = add_extra(dn_items, extra=extra)
    text_dp = join(path_parser.summary_text, '-'.join(dn_items))
    return text_dp


def init_text_dp_for_tdqfs(model_name, cos_threshold, n_iter, diversity_param_tuple, length_budget_tuple, extra):
    text_dp = get_text_dp_for_tdqfs(model_name=model_name,
        cos_threshold=cos_threshold,
        diversity_param_tuple=diversity_param_tuple,
        length_budget_tuple=length_budget_tuple,
        n_iter=n_iter,
        extra=extra)

    if exists(text_dp):
        raise ValueError('text_dp exists: {}'.format(text_dp))
    os.mkdir(text_dp)
    return text_dp


def init_text_dp(model_name, cos_threshold, n_iter, diversity_param_tuple, extra):
    text_dp = get_text_dp(model_name=model_name,
                          cos_threshold=cos_threshold,
                          diversity_param_tuple=diversity_param_tuple,
                          n_iter=n_iter,
                          extra=extra)

    if exists(text_dp):
        raise ValueError('text_dp exists: {}'.format(text_dp))
    os.mkdir(text_dp)

    return text_dp


def get_passage_fps(cid, retrieved_dp, passage_ids=None):
    """

    :param cid:
    :param retrieved_dp:
    :param passage_ids:
    :return:
    """
    year, _ = cid.split(config.SEP)
    cc_dp = join(path_parser.data_passages, year, cid)

    if not passage_ids:
        if retrieved_dp:  # get passage_ids from retrieval file
            if not exists(retrieved_dp):
                raise ValueError('retrieved_dp does not exist: {}'.format(retrieved_dp))

            fp = join(retrieved_dp, cid)
            with io.open(fp, encoding='utf-8') as f:
                content = f.readlines()

            passage_ids = [ll.rstrip('\n').split('\t')[0] for ll in content]

        else:  # get passage_ids from data dir
            passage_ids = [pid for pid in listdir(cc_dp) if isfile(join(cc_dp, pid))]

    passage_fps = [join(cc_dp, pid) for pid in passage_ids]

    return passage_ids, passage_fps


def get_passage_fps_for_tdqfs(cid, retrieved_dp, passage_ids=None):
    """

    :param cid:
    :param retrieved_dp:
    :param passage_ids:
    :return:
    """
    cc_dp = join(path_parser.data_tdqfs_passages, cid)

    if not passage_ids:
        if retrieved_dp:  # get passage_ids from retrieval file
            if not exists(retrieved_dp):
                raise ValueError('retrieved_dp does not exist: {}'.format(retrieved_dp))

            fp = join(retrieved_dp, cid)
            with io.open(fp, encoding='utf-8') as f:
                content = f.readlines()

            passage_ids = [ll.rstrip('\n').split('\t')[0] for ll in content]

        else:  # get passage_ids from data dir
            passage_ids = [pid for pid in listdir(cc_dp) if isfile(join(cc_dp, pid))]

    passage_fps = [join(cc_dp, pid) for pid in passage_ids]
    
    return passage_ids, passage_fps


def get_query_w_cid(query_info, cid):
    return query_info[cid]
