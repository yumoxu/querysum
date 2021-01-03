import io
import sys
from os import mkdir, listdir
from os.path import join, dirname, abspath, exists
import json
from random import choice
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import itertools
from multiprocessing import Pool

sys_path = dirname(dirname(abspath(__file__)))
parent_sys_path = dirname(sys_path)

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)


data_root = '~/querysum/data/tdqfs'


def build_summary_targets():
    src_dp = join(data_root, 'manual_summaries')
    cids = [dn for dn in listdir(src_dp)]

    dump_dp = join(data_root, 'summary_targets')
    for cid in tqdm(cids):
        src_cc_dp = join(src_dp, cid)
        fns = [fn for fn in listdir(src_cc_dp)]

        dump_cc_dp = join(dump_dp, cid)
        mkdir(dump_cc_dp)
        for fn in fns:
            summ = io.open(join(src_cc_dp, fn)).read().strip('\n')
            sentences = sent_tokenize(summ)
            
            proc_sentences = []
            for ss in sentences:
                ss = ss.strip().strip('\n')
                if ss:
                    proc_sentences.append(ss)
            assert proc_sentences
            io.open(join(dump_cc_dp, fn), mode='a').write('\n'.join(proc_sentences))


def read_data(json_fp):
    with open(json_fp) as f:
        data = json.load(f)
    questions = []
    answers = []
    supports = []
    for d in data:
        questions.append(d["question"].strip())
        answers.append(d["answer"].strip())
        supports.append(d["document"].strip())
    assert(len(questions) == len(answers) == len(supports))

    index = choice(range(len(questions)))
    # print(f'loaded {len(questions)} samples!\n======Question======\nq: {questions[index]}\n======Answer======\na: {answers[index]}\n======Document======\nd: {supports[index]}')
    return questions, answers, supports


def build_question_files(questions, data_type, simplified=False):
    lines = []
    for idx, qq in enumerate(tqdm(questions)):
        if simplified:
            assert qq.split('--T--')[0].strip(), f'empty q after proc: {qq}'
            qq = qq.split('--T--')[0].strip()

        cid = f'{data_type}_{idx}'
        record = f'{cid}\t{qq}'
        lines.append(record)
    
    if simplified:
        fn = f'{data_type}_question_simplified.txt'
    else:
        fn = f'{data_type}_question_raw.txt'

    io.open(join(data_root, fn), mode='a').write('\n'.join(lines))


def build_answer_files(answers, data_type):
    dp = join(data_root, f'{data_type}_answers')
    if not exists(dp):
        mkdir(dp)
    for idx, ans in enumerate(tqdm(answers)):
        cid = f'{data_type}_{idx}'
        io.open(join(dp, f'{cid}.txt'), mode='a').write(ans)


def build_document_files(documents, data_type):
    dp = join(data_root, f'{data_type}_documents')
    if not exists(dp):
        mkdir(dp)
    for idx, doc in enumerate(tqdm(documents)):
        cid = f'{data_type}_{idx}'
        io.open(join(dp, f'{cid}.txt'), mode='a').write(doc)


def dump_segments(fp, segs, cid):
    lines = [f'{"_".join((cid, str(idx)))}\t{seg}' for idx, seg in enumerate(segs)]
    io.open(fp, mode='a').write('\n'.join(lines))


def proc_doc_into_sentences(data_type):
    doc_dp = join(data_root, f'{data_type}_documents')
    assert exists(doc_dp), f'{doc_dp} does not exist!'
    cids = [fn.split('.')[0] for fn in listdir(doc_dp)]

    sentence_dp = join(data_root, f'{data_type}_sentences')
    if not exists(sentence_dp):
        mkdir(sentence_dp)

    passage_dp = join(data_root, f'{data_type}_passages')
    if not exists(passage_dp):
        mkdir(passage_dp)

    for idx, cid in enumerate(tqdm(cids)):
        fn = f'{cid}.txt'
        doc = io.open(join(doc_dp, fn)).read()
        
        passages = doc.split('<P>')
        passages = [psg.strip() for psg in passages if psg.strip()]
        sentences = [sent_tokenize(psg) for psg in passages]
        dump_segments(join(passage_dp, fn), segs=passages, cid=cid)
        dump_segments(join(sentence_dp, fn), segs=list(itertools.chain(*sentences)), cid=cid)


def build_summary_files(answers, data_type):
    dp = join(data_root, f'{data_type}_summaries')
    if not exists(dp):
        mkdir(dp)
    for idx, ans in enumerate(tqdm(answers)):
        cid = f'{data_type}_{idx}'
        io.open(join(dp, f'{cid}.txt'), mode='a').write(ans)


def _proc_answer_into_summaries(cid, ans_dp, summary_dp):
    """for multiproc"""
    doc = io.open(join(ans_dp, f'{cid}.txt')).read().strip('\n')
    sentences = sent_tokenize(doc)
    io.open(join(summary_dp, cid), mode='a').write('\n'.join(sentences))

    
def proc_answer_into_summaries(data_type):
    ans_dp = join(data_root, f'{data_type}_answers')
    assert exists(ans_dp), f'{ans_dp} does not exist!'
    cids = [fn.split('.')[0] for fn in listdir(ans_dp)]

    summary_dp = join(data_root, f'{data_type}_summaries')
    if not exists(summary_dp):
        mkdir(summary_dp)
    for cid in tqdm(cids):
        doc = io.open(join(ans_dp, f'{cid}.txt')).read().strip('\n')
        sentences = sent_tokenize(doc)
        io.open(join(summary_dp, cid), mode='a').write('\n'.join(sentences))


if __name__ == "__main__":
    build_summary_targets()
