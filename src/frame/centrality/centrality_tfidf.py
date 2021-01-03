import sys
from os.path import join, dirname, abspath

sys_path = dirname(dirname(abspath(__file__)))
parent_sys_path = dirname(sys_path)

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

import numpy as np
from tqdm import tqdm

import utils.config_loader as config
from utils.config_loader import logger, path_parser
import utils.graph_io as graph_io
import utils.graph_tools as graph_tools
import tools.tfidf_tools as tfidf_tools
import tools.general_tools as general_tools
import summ.compute_rouge as rouge
import frame.centrality.centrality_config as centrality_config

MODEL_NAME = 'centrality-tfidf-2007-{}_damp-{}_link'.format(centrality_config.DAMP,
                                                            centrality_config.LINK_TYPE)

if centrality_config.LINK_TYPE == 'inter':
    mask_intra = True
else:
    mask_intra = False


def _build_components(cid, query):
    sim_items = tfidf_tools.build_sim_items_e2e(cid, query, mask_intra=mask_intra)
    rel_scores = sim_items['rel_scores']
    sim_mat = sim_items['doc_sim_mat']
    processed_sents = sim_items['processed_sents']

    rel_vec = rel_scores / np.sum(rel_scores)  # l1 norm to make a distribution

    np.fill_diagonal(sim_mat, 0.0)  # avoid self-transition
    logger.info('sim_mat: {}'.format(sim_mat))

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
                                                                    concat_title_narr=False,
                                                                    query_type=centrality_config.QUERY_TYPE)

    for params in tqdm(test_cid_query_dicts):
        components = _build_components(**params)

        graph_io.dump_sim_mat(sim_mat=components['sim_mat'], sim_mat_dp=sim_mat_dp, cid=params['cid'])
        graph_io.dump_rel_vec(rel_vec=components['rel_vec'], rel_vec_dp=rel_vec_dp, cid=params['cid'])
        graph_io.dump_sid2abs(sid2abs=components['sid2abs'], sid2abs_dp=sid2abs_dp, cid=params['cid'])

        logger.info('[BUILD GRAPH COMPONENT] dumping sim mat file to: {0}'.format(sim_mat_dp))
        logger.info('[BUILD GRAPH COMPONENT] dumping rel vec file to: {0}'.format(rel_vec_dp))
        logger.info('[BUILD GRAPH COMPONENT] dumping sid2abs file to: {0}'.format(sid2abs_dp))


def score_e2e():
    damp = centrality_config.DAMP
    use_rel_vec = True
    if damp == 1.0:
        damp = 0.85
        use_rel_vec = False

    graph_tools.score_end2end(model_name=MODEL_NAME,
                              damp=damp,
                              use_rel_vec=use_rel_vec)


def rank_e2e(omega=10, max_n_iter=100):
    graph_tools.rank_end2end(model_name=MODEL_NAME, omega=omega, max_n_iter=max_n_iter)


def select_e2e(omega=10):
    graph_tools.select_end2end(model_name=MODEL_NAME, omega=omega)


def compute_rouge(omega=10):
    params = {
        'model_name': MODEL_NAME,
        'n_iter': None,
        'cos_threshold': 1.0,  # do not pos cosine similarity criterion
        'omega': omega,
        'manual': True,
    }
    rouge.compute_rouge_end2end(**params)


def search_optimum_omega(max_n_iter=100):
    out_fp = join(path_parser.rouge, 'omega_search-{}'.format(MODEL_NAME))

    for omega in range(11, 21):
        graph_tools.rank_end2end(model_name=MODEL_NAME, omega=omega, max_n_iter=max_n_iter)
        graph_tools.select_end2end(model_name=MODEL_NAME, omega=omega, save_out_fp=out_fp)


if __name__ == '__main__':
    build_components_e2e()
    score_e2e()
    rank_e2e()
    # select_e2e()
    # compute_rouge()
