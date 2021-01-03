import utils.config_loader as config
import frame.ir.ir_config as ir_config

# set following macro vars
QUERY_TYPE = config.NARR

# IR configs: the method should be consistent
IR_MODEL_NAME = ir_config.IR_MODEL_NAME_TF  # for building sid2sent for contextual QA models
# IR_RECORDS_DIR_NAME: sentence lookup, IR_MODEL_NAME_TF (full), IR_RECORDS_DIR_NAME_TF (retrieved)
IR_RECORDS_DIR_NAME = ir_config.IR_MODEL_NAME_TF

if config.meta_model_name == 'bert_qa':
    BERT_TYPE = 'bert'
elif config.meta_model_name == 'bert_base':
    BERT_TYPE = 'bert_base'
elif config.meta_model_name == 'bert_passage':
    BERT_TYPE = 'bert_passage-{}'.format(config.bert_passage_iter)
else:
    raise ValueError('Invalid mode_name: {}'.format(config.meta_model_name))

QA_MODEL_NAME_BERT = 'qa-{}-{}-{}'.format(BERT_TYPE, QUERY_TYPE, IR_RECORDS_DIR_NAME)

RELEVANCE_SCORE_DIR_NAME = QA_MODEL_NAME_BERT

# filter config
FILTER = 'topK'  # topK, conf

CONF_THRESHOLD_QA = 0.95
TOP_NUM_QA =90  # 90: sentence, 110: passage

if FILTER == 'conf':
    FILTER_VAR = CONF_THRESHOLD_QA
elif FILTER == 'topK':
    FILTER_VAR = TOP_NUM_QA
else:
    raise ValueError('Invalid FILTER: {}'.format(FILTER))

QA_RECORD_DIR_NAME_PATTERN = 'qa_records-{}-{}_qa_{}'  # model name, conf
QA_RECORD_DIR_NAME_BERT = QA_RECORD_DIR_NAME_PATTERN.format(QA_MODEL_NAME_BERT, FILTER_VAR, FILTER)

QA_TUNE_DIR_NAME_PATTERN = 'qa_tune-{}'
QA_TUNE_DIR_NAME_BERT = QA_TUNE_DIR_NAME_PATTERN.format(QA_MODEL_NAME_BERT)
