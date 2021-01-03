# -*- coding: utf-8 -*-
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import copy
from data.dataset_parser import dataset_parser
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from frame.ir.ir_tools import load_retrieved_sentences, load_retrieved_passages
import nltk

import utils.config_loader as config
from utils.config_loader import logger
from lexrank import STOPWORDS, LexRank


def get_counts(sents):
    count_vec = CountVectorizer()
    counts = count_vec.fit_transform(sents)
    return counts


def get_tf_idf_mat(sents):
    counts = get_counts(sents)
    tf_idf_transformer = TfidfTransformer()
    tf_idf_mat = tf_idf_transformer.fit_transform(counts)
    return tf_idf_mat


def get_tf_mat(sents):
    counts = get_counts(sents)
    tf_transformer = TfidfTransformer(use_idf=False)
    tf_mat = tf_transformer.fit_transform(counts)
    return tf_mat


def build_cross_document_mask(processed_sents):
    """

    :param processed_sents: 2d lists, docs => sents
    :return:
    """
    n_doc = len(processed_sents)
    n_sents = [len(doc_sents) for doc_sents in processed_sents]
    total_n_sent = sum(n_sents)

    start = 0
    submasks = []
    for doc_idx in range(n_doc):
        doc_sents = processed_sents[doc_idx]
        n_doc_sents = len(doc_sents)
        mask = np.ones([n_doc_sents, total_n_sent], dtype=int)
        end = start + n_doc_sents
        mask[:, start:end] = 0
        start = end
        submasks.append(mask)

    mask = np.concatenate(submasks, axis=0)
    return mask


def _compute_rel_scores_tf(processed_sents, query):
    """

    :param processed_sents:
    :param query_sents:
    :param mask_intra:
    :return:
    """
    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list
    doc_sents = copy.deepcopy(doc_sents)  # avoid affecting the original doc_sents list
    doc_sents.append(query)

    tf_mat = get_tf_mat(sents=doc_sents)

    doc_sent_mat = tf_mat[:-1]
    query_mat = tf_mat[-1]
    doc_query_sim_mat = cosine_similarity(doc_sent_mat, query_mat)

    rel_scores = np.squeeze(doc_query_sim_mat, axis=-1)

    return rel_scores


def _compute_rel_scores_tf_dot(processed_sents, query):
    """

    :param processed_sents:
    :param query_sents:
    :param mask_intra:
    :return:
    """
    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list
    doc_sents = copy.deepcopy(doc_sents)  # avoid affecting the original doc_sents list
    # logger.info('doc_sents: {}'.format(len(doc_sents)))
    doc_sents.append(query)

    tf_mat = get_tf_mat(sents=doc_sents).toarray()
    # logger.info('tf_idf_mat: {}'.format(tf_idf_mat))

    # doc_sent_mat = tf_mat[:-1].A
    # query_mat = tf_mat[-1].A.reshape(-1, 1)
    # logger.info('doc_sent_mat: {}, query_mat: {}'.format(doc_sent_mat.shape, query_mat.shape))
    # logger.info('doc_sent_mat: {}'.format(type(doc_sent_mat)))
    rel_scores = np.matmul(tf_mat[:-1], tf_mat[-1])
    # rel_scores = np.squeeze(doc_query_sim_mat, axis=-1)

    logger.info('rel_scores: {}'.format(rel_scores.shape))

    return rel_scores


def _compute_rel_scores_count(processed_sents, query):
    """
    :param processed_sents:
    :param query_sents:
    :param mask_intra:
    :return:
    """
    # todo: recheck and rerun
    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list
    doc_words = [nltk.tokenize.word_tokenize(sent) for sent in doc_sents]
    query_words = nltk.tokenize.word_tokenize(query)

    rel_scores = []
    for sent_words in doc_words:
        count = sum([sent_words.count(q_w) for q_w in query_words])
        rel_scores.append(count)

    rel_scores = np.array(rel_scores)

    return rel_scores


def _compute_sim_mat_tfidf(processed_sents, query, mask_intra):
    """

    :param processed_sents:
    :param query_sents:
    :param mask_intra:
    :return:
    """
    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list
    doc_sents = copy.deepcopy(doc_sents)  # avoid affecting the original doc_sents list
    doc_sents.append(query)

    tf_idf_mat = get_tf_idf_mat(sents=doc_sents)

    doc_sent_mat = tf_idf_mat[:-1]
    query_mat = tf_idf_mat[-1]
    doc_query_sim_mat = cosine_similarity(doc_sent_mat, query_mat)
    doc_sim_mat = cosine_similarity(doc_sent_mat)

    if mask_intra:
        inter_mask = build_cross_document_mask(processed_sents)
        doc_sim_mat *= inter_mask

    rel_scores = np.squeeze(doc_query_sim_mat, axis=-1)
    res = {
        'rel_scores': rel_scores,
        'doc_sim_mat': doc_sim_mat,
    }
    return res


def build_rel_scores_tf(cid,
                        query,
                        max_ns_doc=None,
                        retrieved_dp=None):
    if retrieved_dp:
        original_sents, processed_sents = load_retrieved_sentences(retrieved_dp=retrieved_dp,
                                                                   cid=cid)
    else:
        original_sents, processed_sents = dataset_parser.cid2sents(cid,
                                                                   max_ns_doc=max_ns_doc)  # 2d lists, docs => sents
    rel_scores = _compute_rel_scores_tf(processed_sents, query)

    res = {
        'rel_scores': rel_scores,
        'processed_sents': processed_sents,
        'original_sents': original_sents,
    }

    return res


def build_rel_scores_tf_passage(cid,
                                query,
                                retrieved_dp=None,
                                tdqfs_data=False):
    _, proc_passages, passage_ids = load_retrieved_passages(cid,
        get_sents=False,
        retrieved_dp=retrieved_dp,
        passage_ids=None,
        tdqfs_data=tdqfs_data)
    # passage_ids, passage_fps = get_passage_fps(cid, retrieved_dp=retrieved_dp)
    #     #
    #     # proc_passages = []
    # # passage_ids = []
    # for fp in passage_fps:
    #     # po = load_obj(fp)
    #     with open(fp, 'rb') as f:
    #         po = dill.load(f)
    #     # print('po: {}, type(po): {}'.format(po, type(po)))
    #     # passage_ids.append(po.pid)
    #
    #     passage = po.get_proc_passage()
    #     proc_passages.append(passage)
    #     # logger.info('{}: {}'.format(po.pid, passage))
    rel_scores = _compute_rel_scores_tf([proc_passages], query)  # nest proc_passages again for compatibility; todo: double check the nest level
    logger.info('rel_scores: {}'.format(rel_scores))

    pid2score = {}
    for pid, score in zip(passage_ids, rel_scores):
        pid2score[pid] = score

    return pid2score


def build_sim_items_e2e(cid,
                        query,
                        mask_intra,
                        max_ns_doc=None,
                        retrieved_dp=None,
                        sentence_rep='tfidf',
                        rm_dialog=True):
    if retrieved_dp:
        original_sents, processed_sents = load_retrieved_sentences(retrieved_dp=retrieved_dp, cid=cid)
    else:
        original_sents, processed_sents = dataset_parser.cid2sents(cid,
                                                                   rm_dialog=rm_dialog,
                                                                   max_ns_doc=max_ns_doc)  # 2d lists, docs => sents

    
    assert sentence_rep == 'tfidf'
    res = _compute_sim_mat_tfidf(processed_sents=processed_sents,
        query=query, mask_intra=mask_intra)

    sim_items = {
        'doc_sim_mat': res['doc_sim_mat'],
        'rel_scores': res['rel_scores'],
        'processed_sents': processed_sents,
        'original_sents': original_sents,
    }

    return sim_items


def build_sim_items_e2e_tfidf_with_lexrank(cid, query, max_ns_doc=None, retrieved_dp=None, rm_dialog=True):
    """
        Initialize LexRank with document-wise organized sentences to get true IDF.

    :param cid:
    :param query:
    :param max_ns_doc:
    :param retrieved_dp:
    :param rm_dialog:
    :return:
    """
    if retrieved_dp:
        original_sents, processed_sents = load_retrieved_sentences(retrieved_dp=retrieved_dp, cid=cid)
    else:
        if 'tdqfs' in config.test_year:
            original_sents, processed_sents = dataset_parser.cid2sents_tdqfs(cid)
        else:
            original_sents, processed_sents = dataset_parser.cid2sents(cid,
                rm_dialog=rm_dialog,
                max_ns_doc=max_ns_doc)  # 2d lists, docs => sents

    lxr = LexRank(processed_sents, stopwords=STOPWORDS['en'])

    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list
    doc_sents = copy.deepcopy(doc_sents)  # avoid affecting the original doc_sents list
    doc_sents.append(query)

    sim_mat = lxr.get_tfidf_similarity_matrix(sentences=doc_sents)

    doc_sim_mat  = sim_mat[:-1, :-1]
    rel_scores = sim_mat[-1, :-1]
    # logger.info('doc_sim_mat: {}, rel_scores: {}'.format(doc_sim_mat.shape, rel_scores.shape))

    sim_items = {
        'doc_sim_mat': doc_sim_mat,
        'rel_scores': rel_scores,
        'processed_sents': processed_sents,
        'original_sents': original_sents,
    }

    return sim_items

