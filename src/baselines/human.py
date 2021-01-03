import sys
import os
from os import listdir
from os.path import join, dirname, abspath, isfile, exists
sys_path = dirname(dirname(abspath(__file__)))
parent_sys_path = dirname(sys_path)

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

import shutil
import utils.config_loader as config
from utils.config_loader import logger, path_parser
from pyrouge import Rouge155

"""
    This module computes Gold scores which represents human intra-agreement.
"""

MODEL_DP = join(path_parser.data_summary_targets, config.test_year)
MODEL_NAME_TEMP = 'human_model'
SYSTEM_NAME_TEMP = 'human_system'

MODEL_DP_TEMP = join(path_parser.summary_text, MODEL_NAME_TEMP)
SYSTEM_DP_TEMP = join(path_parser.summary_text, SYSTEM_NAME_TEMP)
fns = [fn for fn in listdir(MODEL_DP) if isfile(join(MODEL_DP, fn))]
ROUGE_METRICS = ['1', '2', 'SU4']
N_REFS = 4

def build_eval_dirs(summary_index):
    system_fns = []
    model_fns = []
    for fn in fns:
        if fn.endswith(str(summary_index)):
            system_fns.append(fn)
        else:
            model_fns.append(fn)

    assert len(model_fns)/len(system_fns) == N_REFS-1

    # remove previous output
    for temp_dp in (MODEL_DP_TEMP, SYSTEM_DP_TEMP):
        if exists(temp_dp):
            shutil.rmtree(temp_dp)
        os.mkdir(temp_dp)

    for fn in system_fns:
        shutil.copyfile(join(MODEL_DP, fn), join(SYSTEM_DP_TEMP, fn[:-2]))

    for fn in model_fns:
        shutil.copyfile(join(MODEL_DP, fn), join(MODEL_DP_TEMP, fn))

def proc_output(output):
    start_pat = '1 ROUGE-{} Average'

    output = '\n'.join(output.split('\n')[1:])
    inter_breaker = '\n---------------------------------------------\n'
    intra_breaker = '\n.............................................\n'

    tg2ck = {}
    for ck in output.split(inter_breaker):
        ck = ck.strip('\n')
        if ck:
            ck = ck.split(intra_breaker)[0]
            for tg in ROUGE_METRICS:
                if ck.startswith(start_pat.format(tg)):
                    tg2ck[tg] = ck
                    break

    num_idx = 3
    tg2recall = {}
    tg2f1 = {}

    for tg, ck in tg2ck.items():
        lines = ck.split('\n')
        recall = float(lines[0].split(' ')[num_idx]) * 100
        f1 = float(lines[2].split(' ')[num_idx]) * 100

        tg2recall[tg] = recall
        tg2f1[tg] = f1

    return tg2recall, tg2f1

def compute_rouge_for_human(system_dp, model_dp):
    rouge_args = '-a -l 250 -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {} -x'.format(
        path_parser.rouge_dir)
    
    r = Rouge155(rouge_args=rouge_args)
    r.system_dir = system_dp
    r.model_dir = model_dp

    gen_sys_file_pat = '(\w*)'
    gen_model_file_pat = '#ID#_[\d]'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    output = r.convert_and_evaluate()
    tg2recall, tg2f1 = proc_output(output)
    return tg2recall, tg2f1

def compute_rouge():
    tg2recall_all = {}
    tg2f1_all = {}
    for idx in range(N_REFS):
        build_eval_dirs(summary_index=idx+1)
        tg2recall, tg2f1 = compute_rouge_for_human(system_dp=SYSTEM_DP_TEMP, model_dp=MODEL_DP_TEMP)
        logger.info('tg2f1: {}'.format(tg2f1))
        for metric in ROUGE_METRICS:
            if metric in tg2recall_all:
                tg2recall_all[metric] += tg2recall[metric]
            else:
                tg2recall_all[metric] = tg2recall[metric]

            if metric in tg2f1_all:
                tg2f1_all[metric] += tg2f1[metric]
            else:
                tg2f1_all[metric] = tg2f1[metric]

    for metric in ROUGE_METRICS:
        tg2recall_all[metric] /= N_REFS
        tg2f1_all[metric] /= N_REFS

    recall_str = 'Recall:\t{}'.format('\t'.join(['{0:.2f}'.format(tg2recall_all[metric]) for metric in ROUGE_METRICS]))
    f1_str = 'F1:\t{}'.format('\t'.join(['{0:.2f}'.format(tg2f1_all[metric]) for metric in ROUGE_METRICS]))

    output = '\n' + '\n'.join((f1_str, recall_str))
    logger.info(output)

if __name__ == '__main__':
    compute_rouge()
