import sys
import io
import os
from os import listdir
from os.path import join, dirname, abspath, isfile, exists, isdir
sys_path = dirname(dirname(abspath(__file__)))
parent_sys_path = dirname(sys_path)

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

from utils.config_loader import logger, path_parser
import summ.compute_rouge as rouge

MODEL_NAME = 'lead-tqdfs'
cids = [cid for cid in listdir(path_parser.data_tdqfs_sentences) if isdir(join(path_parser.data_tdqfs_sentences, cid))]

text_dp = join(path_parser.summary_text, MODEL_NAME)
assert not exists(text_dp), f'{text_dp} exists!'
os.mkdir(text_dp)


def _get_lines(cid):
    sents = []
    for doc_idx in range(10):
        if len(sents) >= 50:
            return sents
        fp = join(path_parser.data_tdqfs_sentences, cid, str(doc_idx))
        lines = [line.strip('\n') for line in io.open(fp).readlines()]
        sents.extend([line for line in lines if line])
    return sents

def select():
    for cid in cids:
        lines = _get_lines(cid)
        logger.info(f'{cid}: {len(lines)}')
        io.open(join(text_dp, cid), mode='a').write('\n'.join(lines))


def compute_rouge():
    rouge_parmas = {
        'text_dp': text_dp,
        'ref_dp': path_parser.data_tdqfs_summary_targets,
        'length': 250,
    }
    output = rouge.compute_rouge_for_tdqfs(**rouge_parmas)
    return output


if __name__ == "__main__":
    select()
    compute_rouge()
