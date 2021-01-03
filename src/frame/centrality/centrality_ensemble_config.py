import utils.config_loader as config
import frame.bert_ensemble.ensemble_config as ensemble_config

# set following macro vars
QUERY_TYPE = config.NARR
LINK_TYPE = 'uniform'  # inter, intra, uniform

QA_RECORD_DIR_NAME = ensemble_config.QA_RECORD_DIR_NAME  # sentence lookup for using hard filtering

DAMP = 0.85 # 1.0, 0.85

DIVERSITY_ALGORITHM = 'wan'  # wan
OMEGA = 0  # 4 (sentence), 2 (passage)
COS_THRESHOLD = 0.6  # 0.5, 0.6, 1.0

if DIVERSITY_ALGORITHM == 'wan':
    DIVERSITY_PARAM_TUPLE = (OMEGA, DIVERSITY_ALGORITHM)
else:
    raise ValueError('Invalid DIVERSITY_ALGORITHM: {}'.format(DIVERSITY_ALGORITHM))

QA_SCORE_DIR_NAME = QA_RECORD_DIR_NAME
BIAS_TYPE = 'hard'  # hard, soft

CENTRALITY_MODEL_NAME_BASIC = 'centrality-{}_bias-{}_damp-{}'.format(BIAS_TYPE, DAMP, QA_SCORE_DIR_NAME)
CENTRALITY_TUNE_DIR_NAME_BASIC = 'centrality_tune-{0}-{1}_cos-{2}'.format(CENTRALITY_MODEL_NAME_BASIC,
    COS_THRESHOLD, DIVERSITY_ALGORITHM)

# LENGTH_BUDGET_TUPLE = ('ns', 7)
LENGTH_BUDGET_TUPLE = ('nw', 250)
