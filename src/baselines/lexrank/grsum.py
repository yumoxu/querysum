# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname, abspath, exists

sys_path = dirname(dirname(abspath(__file__)))
parent_sys_path = dirname(sys_path)

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

from tqdm import tqdm
import utils.config_loader as config
from utils.config_loader import logger, config_meta
import utils.graph_io as graph_io
import utils.graph_tools as graph_tools

import summ.select_sent as select_sent
import tools.tfidf_tools as tfidf_tools
import tools.general_tools as general_tools
import tools.vec_tools as vec_tools

# MODEL_NAME = 'grsum-{}'.format(config.test_year)
MODEL_NAME = 'grsum-{}'.format(config_meta['test_year'])
DIVERSITY_PARAM_TUPLE = (10, 'wan')
COS_THRESHOLD = 1.0
DAMP = 0.85
RM_DIALOG = False

QUERY_TYPE = None  # config.NARR, config.TITLE, None
CONCAT_TITLE_NARR = False if QUERY_TYPE else True

def _build_components(cid, query):
    sim_items = tfidf_tools.build_sim_items_e2e_tfidf_with_lexrank(cid, query, rm_dialog=RM_DIALOG)

    sim_mat = vec_tools.norm_sim_mat(sim_mat=sim_items['doc_sim_mat'], max_min_scale=False)
    rel_vec = vec_tools.norm_rel_scores(rel_scores=sim_items['rel_scores'], max_min_scale=False)
    logger.info('rel_vec: {}'.format(rel_vec))

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
        'model_name': MODEL_NAME,
        'n_iter': None,
        'mode': 'w',
    }

    summ_comp_root = graph_io.get_summ_comp_root(**dp_params)
    sim_mat_dp = graph_io.get_sim_mat_dp(summ_comp_root, mode='w')
    rel_vec_dp = graph_io.get_rel_vec_dp(summ_comp_root, mode='w')
    sid2abs_dp = graph_io.get_sid2abs_dp(summ_comp_root, mode='w')

    logger.info('sim_mat_dp: {}'.format(sim_mat_dp))
    logger.info('rel_vec_dp: {}'.format(rel_vec_dp))
    logger.info('sid2abs_dp: {}'.format(sid2abs_dp))

    test_cid_query_dicts = general_tools.build_test_cid_query_dicts(tokenize_narr=False,
                                                                    concat_title_narr=CONCAT_TITLE_NARR,
                                                                    query_type=QUERY_TYPE)

    for params in tqdm(test_cid_query_dicts):
        logger.info('cid: {}'.format(params['cid']))

        components = _build_components(**params)
        graph_io.dump_sim_mat(sim_mat=components['sim_mat'], sim_mat_dp=sim_mat_dp, cid=params['cid'])
        graph_io.dump_rel_vec(rel_vec=components['rel_vec'], rel_vec_dp=rel_vec_dp, cid=params['cid'])
        graph_io.dump_sid2abs(sid2abs=components['sid2abs'], sid2abs_dp=sid2abs_dp, cid=params['cid'])


def score_e2e():
    if DAMP == 1.0:
        damp = 0.85
        use_rel_vec = False
    else:
        damp = DAMP
        use_rel_vec = True

    graph_tools.score_end2end(model_name=MODEL_NAME,
                              damp=damp,
                              use_rel_vec=use_rel_vec,
                              rm_dialog=RM_DIALOG)


def rank_e2e():
    graph_tools.rank_end2end(model_name=MODEL_NAME,
                             diversity_param_tuple=DIVERSITY_PARAM_TUPLE,
                             retrieved_dp=None,
                             rm_dialog=RM_DIALOG)


def select_e2e():
    params = {
        'model_name': MODEL_NAME,
        'diversity_param_tuple': DIVERSITY_PARAM_TUPLE,
        'cos_threshold': COS_THRESHOLD,
        'rm_dialog': RM_DIALOG,
    }
    select_sent.select_end2end(**params)


if __name__ == '__main__':
    build_components_e2e()
    score_e2e()
    rank_e2e()
    select_e2e()
