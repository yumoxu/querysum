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
from utils.config_loader import logger, path_parser
import utils.graph_io as graph_io
import utils.graph_tools as graph_tools
import tools.tfidf_tools as tfidf_tools
import tools.general_tools as general_tools
import tools.vec_tools as vec_tools
import summ.select_sent as select_sent
import summ.compute_rouge as rouge

import frame.centrality.centrality_config as centrality_config
from tqdm import tqdm
import numpy as np
from utils.tools import get_test_cc_ids, get_text_dp, get_text_dp_for_tdqfs

"""
    This file is for the ablation study of ``w/o Answering'', a combination of the following components:
        1. Retrieval model: TF; score and filter
        3. Summarization model: MRW (sentence rep: TFIDF); with relevance score from IR

    The relevance vector is produced via loading relevance scores and normalization.
"""

ir_record_dp = join(path_parser.summary_rank, centrality_config.IR_RECORD_DIR_NAME)
model_name = centrality_config.CENTRALITY_MODEL_NAME_wo_QA

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


def _load_rel_scores(cid, ir_record_dp):
    ir_record_fp = join(ir_record_dp, cid)
    ir_records = io.open(ir_record_fp, encoding='utf-8').readlines()
    ir_scores = [float(line.split('\t')[1]) for line in ir_records]
    ir_rel_scores = np.array(ir_scores)
    return ir_rel_scores


def _build_components(cid, query):
    sim_items = tfidf_tools.build_sim_items_e2e(cid,
                                                query,
                                                mask_intra=None,
                                                max_ns_doc=None,
                                                retrieved_dp=ir_record_dp,
                                                sentence_rep='tfidf')

    sim_mat = vec_tools.norm_sim_mat(sim_mat=sim_items['doc_sim_mat'], max_min_scale=False)
    # logger.info('sim_mat: {}'.format(sim_mat))

    rel_scores = _load_rel_scores(cid, ir_record_dp=ir_record_dp)
    rel_vec = vec_tools.norm_rel_scores(rel_scores=rel_scores, max_min_scale=False)
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
        'model_name': model_name,
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
    if centrality_config.DAMP == 1.0:
        damp = 0.85
        use_rel_vec = False
    else:
        damp = centrality_config.DAMP
        use_rel_vec = True

    graph_tools.score_end2end(model_name=model_name,
                              damp=damp,
                              use_rel_vec=use_rel_vec,
                              cc_ids=cc_ids)


def rank_e2e():
    graph_tools.rank_end2end(model_name=model_name,
                             diversity_param_tuple=centrality_config.DIVERSITY_PARAM_TUPLE,
                             retrieved_dp=ir_record_dp,
                             cc_ids=cc_ids)


def select_e2e():
    params = {
        'model_name': model_name,
        'diversity_param_tuple': centrality_config.DIVERSITY_PARAM_TUPLE,
        'cos_threshold': centrality_config.COS_THRESHOLD,  # do not pos cosine similarity criterion?
        'retrieved_dp': ir_record_dp,
    }
    select_sent.select_end2end(**params)


def select_e2e_tdqfs():
    params = {
        'model_name': model_name,
        'n_iter': None,
        'length_budget_tuple': centrality_config.LENGTH_BUDGET_TUPLE,
        'diversity_param_tuple': centrality_config.DIVERSITY_PARAM_TUPLE,
        'cos_threshold': centrality_config.COS_THRESHOLD,  # do not pos cosine similarity criterion?
        'retrieved_dp': ir_record_dp,
        'cc_ids': cc_ids,
    }
    select_sent.select_end2end_for_tdqfs(**params)


def compute_rouge_tdqfs():
    text_params = {
        'model_name': model_name,
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


if __name__ == '__main__':
    build_components_e2e()
    score_e2e()
    rank_e2e()
    # select_e2e()

    select_e2e_tdqfs()
    compute_rouge_tdqfs()
