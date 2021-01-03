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
import re
import utils.config_loader as config
from utils.config_loader import logger, path_parser
import summ.compute_rouge as rouge

sys.path.insert(0, dirname(dirname(abspath(__file__))))

MODEL_NAME = 'lead-{}'.format(config.test_year)

def extract_lead_summaries():
    if config.test_year == '2006':
        dn = '2006/NIST/NISTeval/ROUGE/peers'
    elif config.test_year == '2007':
        dn = '2007/mainEval/ROUGE/peers'
    else:
        raise ValueError('Invalid test_year: {}'.format(config.test_year))

    lead_dp = join(path_parser.data_summary_results, dn)
    fns = [fn for fn in listdir(lead_dp) if isfile(join(lead_dp, fn))]
    ref_pat = re.compile('[\S]+.M.250.\D.1$')
    ref_fns = [fn for fn in fns if re.search(ref_pat, fn)]
    print(ref_fns)

    out_dp = join(path_parser.summary_text, MODEL_NAME)
    if exists(out_dp):
        raise ValueError('out_dp exists: {}'.format(out_dp))
    os.mkdir(out_dp)

    for fn in ref_fns:
        out_fn = '_'.join((config.test_year, fn.split('.')[0] + fn.split('.')[-2]))
        shutil.copyfile(join(lead_dp, fn), join(out_dp, out_fn))

def compute_rouge():
    rouge.compute_rouge(model_name=MODEL_NAME)


if __name__ == '__main__':
    extract_lead_summaries()
    compute_rouge()
