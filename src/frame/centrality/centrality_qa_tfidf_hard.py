import sys
from os.path import join, dirname, abspath, exists

sys_path = dirname(dirname(abspath(__file__)))
parent_sys_path = dirname(sys_path)

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

import io
import utils.config_loader as config
from utils.config_loader import logger, path_parser, config_model
import utils.graph_io as graph_io
import utils.graph_tools as graph_tools
from utils.tools import get_test_cc_ids, get_text_dp, get_text_dp_for_tdqfs

import tools.tfidf_tools as tfidf_tools
import tools.general_tools as general_tools
import tools.vec_tools as vec_tools
import summ.select_sent as select_sent
import summ.compute_rouge as rouge

import frame.centrality.centrality_config as centrality_config
from tqdm import tqdm
import numpy as np

import shutil
import os


"""
    This file is for the combination of the following components:
        1. IR model: TF; score and filter
        2. QA model: Bert-QA; score and filter
        3. Centrality model: MRW (sentence rep: TFIDF); bias: QA

    Different from the soft version, QA model filters sentences as per their scores.

    The relevance vector is produced via loading relevance scores and normalization.
"""

assert centrality_config.BIAS_TYPE == 'hard'
qa_record_dp = join(path_parser.summary_rank, centrality_config.QA_RECORD_DIR_NAME)
use_tdqfs = 'tdqfs' in centrality_config.QA_RECORD_DIR_NAME

if use_tdqfs:
    sentence_dp = path_parser.data_tdqfs_sentences
    query_fp = path_parser.data_tdqfs_queries
    tdqfs_summary_target_dp = path_parser.data_tdqfs_summary_targets

    test_cid_query_dicts = general_tools.build_tdqfs_cid_query_dicts(query_fp=query_fp, proc=True)
    cc_ids = [cq_dict['cid'] for cq_dict in test_cid_query_dicts]
else:
    test_cid_query_dicts = general_tools.build_test_cid_query_dicts(tokenize_narr=False,
                                                                    concat_title_narr=False,
                                                                    query_type=centrality_config.QUERY_TYPE)
    cc_ids = get_test_cc_ids()


def _load_rel_scores(cid, qa_record_dp):
    qa_record_fp = join(qa_record_dp, cid)

    with io.open(qa_record_fp, encoding='utf-8') as f:
        scores = [float(line.split('\t')[1]) for line in f.readlines()]

    rel_scores = np.array(scores)

    return rel_scores


def _build_components(cid, query):
    mask_intra = centrality_config.LINK_TYPE == 'intra'
    sim_items = tfidf_tools.build_sim_items_e2e(cid,
                                                query,
                                                mask_intra=mask_intra,
                                                max_ns_doc=config_model['max_ns_doc'],
                                                retrieved_dp=qa_record_dp)

    sim_mat = vec_tools.norm_sim_mat(sim_mat=sim_items['doc_sim_mat'], max_min_scale=False)
    # logger.info('sim_mat: {}'.format(sim_mat))

    rel_scores = _load_rel_scores(cid, qa_record_dp=qa_record_dp)

    if centrality_config.REL_VEC_NORM == 'sqrt_tanh' and config.grain == 'passage':
        passage_proc = True
    else:
        passage_proc = False
    rel_vec = vec_tools.norm_rel_scores(rel_scores=rel_scores, max_min_scale=False, passage_proc=passage_proc)
    # logger.info('rel_vec: {}'.format(rel_vec))

    if len(rel_vec) != len(sim_mat):
        raise ValueError('Incompatible sim_mat size: {} and rel_vec size: {} for cid: {}'.format(
            sim_mat.shape, rel_vec.shape, cid))

    processed_sents = sim_items['processed_sents']
    sid2abs = {}
    sid_abs = 0
    for doc_idx, doc in enumerate(processed_sents):
        for sent_idx, sent in enumerate(doc):
            sid = config.SEP.join((str(doc_idx), str(sent_idx)))
            sid2abs[sid] = sid_abs
            sid_abs += 1

    components = {
        'sim_mat': sim_mat,
        'rel_vec': rel_vec,
        'sid2abs': sid2abs,
    }

    return components


def build_components_e2e():
    dp_params = {
        'model_name': centrality_config.CENTRALITY_MODEL_NAME_BASIC,
        'n_iter': None,
        'mode': 'w',
    }

    summ_comp_root = graph_io.get_summ_comp_root(**dp_params)
    sim_mat_dp = graph_io.get_sim_mat_dp(summ_comp_root, mode='w')
    rel_vec_dp = graph_io.get_rel_vec_dp(summ_comp_root, mode='w')
    sid2abs_dp = graph_io.get_sid2abs_dp(summ_comp_root, mode='w')

    for params in tqdm(test_cid_query_dicts):
        components = _build_components(**params)
        graph_io.dump_sim_mat(sim_mat=components['sim_mat'], sim_mat_dp=sim_mat_dp, cid=params['cid'])
        graph_io.dump_rel_vec(rel_vec=components['rel_vec'], rel_vec_dp=rel_vec_dp, cid=params['cid'])
        graph_io.dump_sid2abs(sid2abs=components['sid2abs'], sid2abs_dp=sid2abs_dp, cid=params['cid'])
       

def score_e2e():
    damp = centrality_config.DAMP
    use_rel_vec = True
    if damp == 1.0:
        damp = 0.85
        use_rel_vec = False

    graph_tools.score_end2end(model_name=centrality_config.CENTRALITY_MODEL_NAME_BASIC,
                              damp=damp,
                              use_rel_vec=use_rel_vec,
                              cc_ids=cc_ids)


def rank_e2e():
    graph_tools.rank_end2end(model_name=centrality_config.CENTRALITY_MODEL_NAME_BASIC,
                             diversity_param_tuple=centrality_config.DIVERSITY_PARAM_TUPLE,
                             retrieved_dp=qa_record_dp,
                             cc_ids=cc_ids)


def select_e2e():
    # graph_tools.select_end2end(model_name=model_name, omega=omega)
    params = {
        'model_name': centrality_config.CENTRALITY_MODEL_NAME_BASIC,
        'n_iter': None,
        'diversity_param_tuple': centrality_config.DIVERSITY_PARAM_TUPLE,
        'cos_threshold': centrality_config.COS_THRESHOLD,  # do not pos cosine similarity criterion?
        'retrieved_dp': qa_record_dp,
        'cc_ids': cc_ids,
    }
    select_sent.select_end2end(**params)


def compute_rouge():
    params = {
        'model_name': centrality_config.CENTRALITY_MODEL_NAME_BASIC,
        'n_iter': None,
        'cos_threshold': centrality_config.COS_THRESHOLD,  # do not pos cosine similarity criterion
        'diversity_param_tuple': centrality_config.DIVERSITY_PARAM_TUPLE,
        # 'manual': True,
    }
    rouge.compute_rouge_end2end(**params)


def select_e2e_tdqfs():
    params = {
        'model_name': centrality_config.CENTRALITY_MODEL_NAME_BASIC,
        'n_iter': None,
        'length_budget_tuple': centrality_config.LENGTH_BUDGET_TUPLE,
        'diversity_param_tuple': centrality_config.DIVERSITY_PARAM_TUPLE,
        'cos_threshold': centrality_config.COS_THRESHOLD,  # do not pos cosine similarity criterion?
        'retrieved_dp': qa_record_dp,
        'cc_ids': cc_ids,
    }
    select_sent.select_end2end_for_tdqfs(**params)


def compute_rouge_tdqfs():
    text_params = {
        'model_name': centrality_config.CENTRALITY_MODEL_NAME_BASIC,
        'n_iter': None,
        'length_budget_tuple': centrality_config.LENGTH_BUDGET_TUPLE,
        'diversity_param_tuple': centrality_config.DIVERSITY_PARAM_TUPLE,
        'cos_threshold': centrality_config.COS_THRESHOLD,
        'extra': None,
    }

    text_dp = get_text_dp_for_tdqfs(**text_params)

    rouge_parmas = {
        'text_dp': text_dp,
        'ref_dp': tdqfs_summary_target_dp,
    }
    if centrality_config.LENGTH_BUDGET_TUPLE[0] == 'nw':
        rouge_parmas['length'] = centrality_config.LENGTH_BUDGET_TUPLE[1]
    
    output = rouge.compute_rouge_for_tdqfs(**rouge_parmas)
    return output


def tune():
    """
        Tune centrality hyper-parameter, omega.

        Run the following functions before tuning:
        (1) build_components_e2e
        (2) score_e2e

    :return:
    """
    diversity_algorithm = centrality_config.DIVERSITY_ALGORITHM

    if diversity_algorithm == 'wan':
        tune_range = range(0, 21, 1)
    else:
        raise ValueError('Invalid diversity_algorithm: {}'.format(diversity_algorithm))

    centrality_tune_dp_rank = join(path_parser.summary_rank, centrality_config.CENTRALITY_TUNE_DIR_NAME_BASIC)
    centrality_tune_dp_text = join(path_parser.summary_text, centrality_config.CENTRALITY_TUNE_DIR_NAME_BASIC)

    centrality_tune_result_fp = join(path_parser.tune, centrality_config.CENTRALITY_TUNE_DIR_NAME_BASIC)

    rank_prams = {
        'model_name': centrality_config.CENTRALITY_MODEL_NAME_BASIC,
        'retrieved_dp': qa_record_dp,
        'rank_dp': centrality_tune_dp_rank,
    }

    headline = 'Penalty\tRecall\tF1\n'
    open(centrality_tune_result_fp, mode='a', encoding='utf-8').write(headline)

    for penalty in tune_range:
        if exists(centrality_tune_dp_rank):  # remove previous output
            shutil.rmtree(centrality_tune_dp_rank)

        if exists(centrality_tune_dp_text):  # remove previous output
            shutil.rmtree(centrality_tune_dp_text)
        os.mkdir(centrality_tune_dp_text)

        rank_prams['diversity_param_tuple'] = (penalty, diversity_algorithm)

        graph_tools.rank_end2end(**rank_prams)
        performance = select_sent.select_for_dev(rank_dp=centrality_tune_dp_rank,
                                                 text_dp=centrality_tune_dp_text,
                                                 retrieved_dp=qa_record_dp,
                                                 cos_threshold=centrality_config.COS_THRESHOLD)

        # performance = rouge.compute_rouge_for_dev(centrality_tune_dp_text, tune_centrality=True)
        assert  diversity_algorithm == 'wan'
        rec = '{0}\t{1}\n'.format(penalty, performance)
        open(centrality_tune_result_fp, mode='a', encoding='utf-8').write(rec)


if __name__ == '__main__':
    build_components_e2e()
    score_e2e()

    # tune()
    rank_e2e()
    # select_e2e()

    # compute_rouge()
    select_e2e_tdqfs()
    compute_rouge_tdqfs()
