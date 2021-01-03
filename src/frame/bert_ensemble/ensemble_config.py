import sys
from os.path import join, dirname, abspath

sys_path = dirname(dirname(abspath(__file__)))
parent_sys_path = dirname(sys_path)

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

import utils.config_loader as config
from utils.config_loader import path_parser

assert config.grain == 'sent'  # rank and its records are saved under sent directory

SENT_QA_REC_DIR_NAME = f'qa_records-qa-bert-narr-ir_records-ir-tf-{config.test_year}-0.75_ir_conf-90_qa_topK'
PASSAGE_QA_REC_DIR_NAME = f'qa_records-qa-bert_passage-12000-narr-ir_records-ir-passage-tf-{config.test_year}-0.75_ir_conf-90_qa_topK'
SENT_QA_RECORD_DP = join(path_parser.proj_root, 'rank', SENT_QA_REC_DIR_NAME)
PASSAGE_QA_RECORD_DP = join(path_parser.proj_root, 'rank_passage', PASSAGE_QA_REC_DIR_NAME)

# avg: avg two scores; if there is only one score, keep it.
# sqrt; sqrt two scores; if there is only one score, keep it.
# avg_global: avg two scores; if there is only one score, halve it.
# sqrt_global: sqrt two scores; if there is only one score, sqrt it.
# weight_avg: (1- \mu) * sent_score + \mu * span_score
# weight_avg_sent_only: (1- \mu) * sent_score + \mu * span_score; use only records from sent model.
ENSEMBLE_MODE = 'weight_avg_sent_only'
IS_ENSEMBLE_GLOBAL = ENSEMBLE_MODE.endswith('global')
IS_SENT_REC_ONLY = ENSEMBLE_MODE.endswith('sent_only')

MODEL_NAME = 'bert_ensemble-{}-{}_mode'.format(config.test_year, ENSEMBLE_MODE)

FILTER = 'topK'
FILTER_VAR = 90
QA_RECORD_DIR_NAME_PATTERN = 'qa_records-{}-{}_qa_{}'
QA_RECORD_DIR_NAME = QA_RECORD_DIR_NAME_PATTERN.format(MODEL_NAME, FILTER_VAR, FILTER)

SPAN_REC_WEIGHT = 0.0 # 0.05
SPAN_AFFIX = '-{}_span_weight'.format(SPAN_REC_WEIGHT)
if ENSEMBLE_MODE.startswith('weight_avg'):
    MODEL_NAME += SPAN_AFFIX
    QA_RECORD_DIR_NAME += SPAN_AFFIX
