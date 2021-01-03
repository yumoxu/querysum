from os.path import join, dirname, abspath
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from pyrouge import Rouge155
import utils.config_loader as config
from utils.config_loader import path_parser, logger
import utils.tools as tools
import logging


def proc_output_for_tune(output):
    start_pat = '1 ROUGE-2 Average'

    output = '\n'.join(output.split('\n')[1:])
    inter_breaker = '\n---------------------------------------------\n'
    intra_breaker = '\n.............................................\n'

    target_ck = None
    for ck in output.split(inter_breaker):
        ck = ck.strip('\n')
        if ck:
            ck = ck.split(intra_breaker)[0]
            if ck.startswith(start_pat):
                target_ck = ck
                break

    if not target_ck:
        raise ValueError('Not found record!')

    num_idx = 3
    lines = target_ck.split('\n')
    recall = '{0:.2f}'.format(float(lines[0].split(' ')[num_idx]) * 100)
    f1 = '{0:.2f}'.format(float(lines[2].split(' ')[num_idx]) * 100)

    return '\t'.join((recall, f1))


def proc_output(output, target=['1', '2', 'SU4']):
    start_pat = '1 ROUGE-{} Average'

    output = '\n'.join(output.split('\n')[1:])
    inter_breaker = '\n---------------------------------------------\n'
    intra_breaker = '\n.............................................\n'

    tg2ck = {}
    for ck in output.split(inter_breaker):
        ck = ck.strip('\n')
        if ck:
            ck = ck.split(intra_breaker)[0]
            for tg in target:
                if ck.startswith(start_pat.format(tg)):
                    tg2ck[tg] = ck
                    break

    num_idx = 3

    tg2recall = {}
    tg2f1 = {}

    for tg, ck in tg2ck.items():
        lines = ck.split('\n')
        recall = '{0:.2f}'.format(float(lines[0].split(' ')[num_idx]) * 100)
        f1 = '{0:.2f}'.format(float(lines[2].split(' ')[num_idx]) * 100)

        tg2recall[tg] = recall
        tg2f1[tg] = f1

    recall_str = 'Recall:\t{}'.format('\t'.join([tg2recall[tg] for tg in target]))
    f1_str = 'F1:\t{}'.format('\t'.join([tg2f1[tg] for tg in target]))

    output = '\n' + '\n'.join((f1_str, recall_str))
    return output


def compute_rouge_mix(model_name, n_iter, cos_threshold, extra):
    for year in config.years:
        compute_rouge(model_name, n_iter=n_iter,
                      cos_threshold=cos_threshold,
                      year=year,
                      extra=extra)
    return None


def compute_rouge_for_dev(text_dp, tune_centrality):
    rouge_args = '-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {} -x'.format(
        path_parser.rouge_dir)

    if tune_centrality:  # summary length requirement
        rouge_args += ' -l 250'

    r = Rouge155(rouge_args=rouge_args)
    r.system_dir = text_dp
    r.model_dir = join(path_parser.data_summary_targets, config.test_year)

    gen_sys_file_pat = '(\w*)'
    gen_model_file_pat = '#ID#_[\d]'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    output = r.convert_and_evaluate()
    output = proc_output_for_tune(output)
    logger.info(output)
    return output


def compute_rouge_for_ablation_study(text_dp, ref_dp=None):
    rouge_args = '-a -l 250 -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {} -x'.format(
        path_parser.rouge_dir)

    r = Rouge155(rouge_args=rouge_args)
    r.system_dir = text_dp
    if not ref_dp:
        ref_dp = join(path_parser.data_summary_targets, config.test_year)
    r.model_dir = ref_dp

    gen_sys_file_pat = '(\w*)'
    gen_model_file_pat = '#ID#_[\d]'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    output = r.convert_and_evaluate()
    output = proc_output(output)
    logger.info(output)
    return output


def compute_rouge(model_name, n_iter=None, diversity_param_tuple=None, cos_threshold=None, extra=None):
    rouge_args = '-a -l 250 -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {} -x'.format(
        path_parser.rouge_dir)

    r = Rouge155(rouge_args=rouge_args)

    baselines_wo_config = ['lead', 'lead-2006', 'lead-2007', 'lead_2007']
    if model_name in baselines_wo_config or model_name.startswith('duc'):
        text_dp = join(path_parser.summary_text, model_name)
    else:
        text_dp = tools.get_text_dp(model_name,
                                    cos_threshold=cos_threshold,
                                    n_iter=n_iter,
                                    diversity_param_tuple=diversity_param_tuple,
                                    extra=extra)

    r.system_dir = text_dp
    r.model_dir = join(path_parser.data_summary_targets, config.test_year)
    gen_sys_file_pat = '(\w*)'
    gen_model_file_pat = '#ID#_[\d]'

    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    output = r.convert_and_evaluate()
    output = proc_output(output)
    logger.info(output)
    return output


def compute_rouge_end2end(model_name, n_iter, cos_threshold=None, diversity_param_tuple=None, extra=None):
    rouge_paras = {
        'model_name': model_name,
        'n_iter': n_iter,
        'cos_threshold': cos_threshold,
        'diversity_param_tuple': diversity_param_tuple,
        'extra': extra,
    }
    return compute_rouge(**rouge_paras)


def compute_rouge_for_tdqfs(text_dp, ref_dp, length):
    rouge_args = f'-a -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.rouge_dir} -x'
    if length:
        rouge_args += f' -l {length}'
    r = Rouge155(rouge_dir=str(path_parser.remote_root / 'ROUGE-1.5.5'),
                    rouge_args=rouge_args, 
                    log_level=logging.WARNING, 
                    config_parent_dir=str(path_parser.remote_root))


    r.system_dir = text_dp
    r.model_dir = ref_dp

    gen_sys_file_pat = '(\w*)'
    gen_model_file_pat = '#ID#_[\d]'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    output = r.convert_and_evaluate()
    output = proc_output(output)
    logger.info(output)
    return output
