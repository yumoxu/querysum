import utils.config_loader as config
from utils.config_loader import config_meta

if config.grain == 'sent':
    IR_META_NAME = 'ir'
else:
    IR_META_NAME = 'ir-{}'.format(config.grain)  # e.g., ir-passage

QUERY_TYPE = None  # config.NARR, config.TITLE, None (concat narr and title), REF (oracle)
if QUERY_TYPE:
    CONCAT_TITLE_NARR = False
    IR_META_NAME = '{}-{}'.format(IR_META_NAME, QUERY_TYPE)
else:
    CONCAT_TITLE_NARR = True

test_year = config_meta['test_year']
IR_MODEL_NAME_TF = '{}-tf-{}'.format(IR_META_NAME, test_year)

DEDUPLICATE = False

CONF_THRESHOLD_IR = 0.75
TOP_NUM_IR = 90

FILTER = 'conf'  # conf, comp, topK (only used in ablation study)
if FILTER == 'conf':
    FILTER_VAR = CONF_THRESHOLD_IR
elif FILTER == 'topK':
    FILTER_VAR = TOP_NUM_IR
else:
    raise ValueError('Invalid FILTER: {}'.format(FILTER))

IR_RECORDS_DIR_NAME_PATTERN = 'ir_records-{}-{}_ir_{}'

if DEDUPLICATE:
    IR_RECORDS_DIR_NAME_PATTERN += '-dedup'

IR_RECORDS_DIR_NAME_TF = IR_RECORDS_DIR_NAME_PATTERN.format(IR_MODEL_NAME_TF, FILTER_VAR, FILTER)
IR_TUNE_DIR_NAME_TF = 'ir_tune-{}'.format(IR_MODEL_NAME_TF)
