import utils.config_loader as config
import frame.bert_qa.qa_config as qa_config

# set following macro vars
QUERY_TYPE = config.NARR
LINK_TYPE = 'uniform'  # inter, intra, uniform

IR_RECORD_DIR_NAME = qa_config.IR_RECORDS_DIR_NAME  # sentence lookup for using rel_vec

# sentence lookup for using hard filtering
# change to QA_DIR_NAME_BERT when using all sentences (TBC)
QA_RECORD_DIR_NAME = qa_config.QA_RECORD_DIR_NAME_BERT  

QA_RELEVANCE_SCORE_DIR_NAME = qa_config.RELEVANCE_SCORE_DIR_NAME

DAMP = 0.85 # 1.0 (w/o query bias), 0.85 (w/ query bias)

DIVERSITY_ALGORITHM = 'wan'

# for DUC, set it to 4 for sentence, and 2 for passage
# for TD-QFS, we do not use wan's diversity algorithm and set it to 0
OMEGA = 4

COS_THRESHOLD = 0.6

BIAS_TYPE = 'hard'  # hard, soft

if DIVERSITY_ALGORITHM == 'wan':
    DIVERSITY_PARAM_TUPLE = (OMEGA, DIVERSITY_ALGORITHM)
else:
    raise ValueError('Invalid DIVERSITY_ALGORITHM: {}'.format(DIVERSITY_ALGORITHM))

if BIAS_TYPE == 'soft':
    QA_SCORE_DIR_NAME = QA_RELEVANCE_SCORE_DIR_NAME
elif BIAS_TYPE == 'hard':
    QA_SCORE_DIR_NAME = QA_RECORD_DIR_NAME
else:
    raise ValueError('Invalid BIAS_TYPE: {}'.format(BIAS_TYPE))

CENTRALITY_MODEL_NAME_BASIC = 'centrality-{}_bias-{}_damp-{}'.format(BIAS_TYPE, DAMP, QA_SCORE_DIR_NAME)
REL_VEC_NORM = None
if config.grain == 'passage':
    REL_VEC_NORM = 'sqrt_tanh'  # None, sqrt_tanh
    CENTRALITY_MODEL_NAME_BASIC += '-{}_vec_norm'.format(REL_VEC_NORM)


# following is for ablation study of QA module; only IR is used for centrality.
CENTRALITY_MODEL_NAME_wo_QA = 'centrality-{}_bias-{}_damp-{}'.format(BIAS_TYPE, DAMP, IR_RECORD_DIR_NAME)
CENTRALITY_TUNE_DIR_NAME_BASIC = 'centrality_tune-{0}-{1}_cos-{2}'.format(CENTRALITY_MODEL_NAME_BASIC, COS_THRESHOLD, DIVERSITY_ALGORITHM)

# LENGTH_BUDGET_TUPLE = ('ns', 7)
LENGTH_BUDGET_TUPLE = ('nw', 250)
