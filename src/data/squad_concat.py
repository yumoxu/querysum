# -*- coding: utf-8 -*-
"""
Structure of Json File

-data : list (One data object corresponds to one topic in Wikipedia. Paragraph is one paragraph from the article)
    -[paragraphs]
        - context
        - qas
            -[question]
                -id
                -[answers]
                    -text
                    -answer_start
"""

import sys
import io
from os.path import join, dirname, abspath, exists
sys_path = dirname(dirname(abspath(__file__)))
parent_sys_path = dirname(sys_path)

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

import utils.config_loader as config
from utils.config_loader import logger, path_parser, config_model

import json
from nltk import word_tokenize, sent_tokenize
import re
import itertools
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

###########################################################################################
def _digit_to_N(word):
    # 3.3 => N; 3,33.3 =>NNN
    if re.match(r'^(\d+([.]|[,]))*\d+$', word):
        word = re.sub(r'[,]', '', word)
        integer_len = len(re.match(r'^\d+', word).group())
        return 'N'*integer_len if integer_len <= 5 else 'NNNNN'
    # 333333 => NNNNN
    elif re.match(r'^\d+$', word):
        word_len = len(word)
        return 'N'*word_len if word_len <= 5 else 'NNNNN'
    # 3,333 => NNNN
    # elif re.match(r'^(\d+[,])*\d+$', word):
    #     word_len = len(re.sub(r'[,]', '', word))
    #     return 'N' * word_len if word_len <= 5 else 'NNNNN'
    else:
        return word


def _wordlist_corner_case_tokenize(wordlist):
    return list(itertools.chain.from_iterable([_word_corner_case_tokenize(word) for word in wordlist]))


def _seperate_dash_colon(word):
    # re.match(r'^(.*[-].*)\1*')
    return re.sub(r'[:]', ' : ', re.sub(r'-|—|–|_', ' - ', word))


def _seperate_slash_backslash(word):
    return re.sub(r'[/]', ' / ', re.sub(r'[\\]', ' \\ ', word))


def _word_corner_case_tokenize(word):
    word = _seperate_dash_colon(word)
    word = _seperate_slash_backslash(word)
    return [_digit_to_N(w) for w in word.split()]

#################################################################################################

def _tokenize_sent(sentence):
    words = [s.lower() for s in word_tokenize(sentence)]
    return _wordlist_corner_case_tokenize(words)


def tokenize_data(data_type):
    if data_type == 'train':
        raw_fp = join(path_parser.squad_raw, 'train-v2.0.json')
    elif data_type == 'dev':
        raw_fp = join(path_parser.squad_raw, 'dev-v2.0.json')
    else:
        raise ValueError('Invalid data_type: {}'.format(data_type))

    with io.open(raw_fp, encoding='utf-8') as raw_f:
        raw_data = json.load(raw_f)

    ind = 0
    qid = 0
    context_dict = {}
    question_dict = {}
    answer_dict = {}
    words = []
    for article in raw_data['data']:
        for paragraph in article['paragraphs']:
            sentence_lens = []
            tokenized_context = []
            sentences = sent_tokenize(paragraph['context'])
            for sentence in sentences:
                sentence_lens.append(len(sentence))
                sent_words = _tokenize_sent(sentence)
                words.extend(sent_words)
                tokenized_context.append(sent_words)
            context_dict[ind]= tokenized_context

            for questionId in paragraph['qas']:
                question = questionId['question']
                question_words = _tokenize_sent(question)
                words.extend(question_words)
                question_dict[qid] = [question_words, ind]

                answer = questionId['answers'][0]
                answer_start = answer['answer_start']
                char_sum = 0
                for i, nc in enumerate(sentence_lens):
                    char_sum += nc
                    if answer_start < char_sum:
                        answer_dict[qid] = i
                        break
                    char_sum += 1
                qid += 1
            ind += 1

    return context_dict, question_dict, answer_dict, words


def build_basic_info(data_type):
    if data_type == 'train':
        raw_fp = join(path_parser.squad_raw, 'train-v2.0.json')
    elif data_type == 'dev':
        raw_fp = join(path_parser.squad_raw, 'dev-v2.0.json')
    else:
        raise ValueError('Invalid data_type: {}'.format(data_type))

    with io.open(raw_fp, encoding='utf-8') as raw_f:
        raw_data = json.load(raw_f)

    sid2sent = {}
    qid2question = {}  # qid: join(pid, str(q_idx))
    qid2ans_sid = {}
    n_no_ans = 0

    for doc_idx, doc in enumerate(raw_data['data']):
        for para_idx, para in enumerate(doc['paragraphs']):
            pid = config.SEP.join((str(doc_idx), str(para_idx)))
            sid2sent[pid] = {}

            sentence_lens = []
            sentences = sent_tokenize(para['context'])

            for sent_idx, sentence in enumerate(sentences):
                sid = config.SEP.join((pid, str(sent_idx)))
                sentence_lens.append(len(sentence))
                sid2sent[pid][sid] = sentence.replace('\n', ' ')

            for q_idx, qas_item in enumerate(para['qas']):
                qid = config.SEP.join((pid, str(q_idx)))
                qid2question[qid] = qas_item['question']

                if qas_item['is_impossible']:  # not answerable
                    qid2ans_sid[qid] = ''
                    n_no_ans += 1
                    continue

                answer = qas_item['answers'][0]
                answer_start = answer['answer_start']

                char_sum = 0

                for candidate_ans_sent_idx, nc in enumerate(sentence_lens):
                    char_sum += nc
                    if answer_start < char_sum:
                        ans_sid = config.SEP.join((pid, str(candidate_ans_sent_idx)))
                        qid2ans_sid[qid] = ans_sid
                        break
                    char_sum += 1  # todo: test this

    logger.info('n_no_ans: {} / {}'.format(n_no_ans, len(qid2question)))
    return qid2ans_sid, qid2question, sid2sent


def build_rec(yy, qid, sid, question, sent):
    return '\t'.join([str(yy), qid, sid, question, sent]) + '\n'


def build_sent_with_context(sid, pid, sid2sent, window):
    if window < 0:
        raise ValueError('Invalid window: {}'.format(window))

    sid2sent_para = sid2sent[pid]

    sent = sid2sent_para[sid]
    if window == 0:
        return sent

    n_sent = len(sid2sent_para)
    sent_idx = int(sid.split(config.SEP)[-1])
    context = []

    context_idx = 0
    context_token_pat = '[unused{}] '

    for i in range(window):
        # preceding
        idx_a = sent_idx - i - 1
        context_idx += 1
        context_token = context_token_pat.format(context_idx)

        if idx_a >= 0:
            context_sid = config.SEP.join([pid, str(idx_a)])
            context.append(context_token + sid2sent_para[context_sid])
        else:
            context.append(context_token)

        # subsequent
        idx_b = sent_idx + i + 1
        context_idx += 1
        context_token = context_token_pat.format(context_idx)
        if idx_b < n_sent:
            # print('n_sent: {}'.format(n_sent))
            context_sid = config.SEP.join([pid, str(idx_b)])
            context.append(context_token + sid2sent_para[context_sid])
        else:
            context.append(context_token)

    context.insert(0, sent)
    sent = ' '.join(context)
    return sent


def sid2pid(sid):
    return config.SEP.join(sid.split(config.SEP)[:-1])


def dump_core(item, dump_fp, qid2question, sid2sent, window):
    qid, ans_sid = item
    question = qid2question[qid]
    pid_ = sid2pid(sid=qid)
    # logger.info('pid_: {}'.format(pid_))

    pos_rec = None
    if ans_sid:
        if window == 0:
            ans = sid2sent[pid_][ans_sid]
        elif window > 0:
            ans = build_sent_with_context(sid=ans_sid, pid=pid_, sid2sent=sid2sent, window=window)
        else:
            raise ValueError('Invalid window: {}'.format(window))

        pos_rec = build_rec(yy=1, qid=qid, sid=ans_sid, question=question, sent=ans)

    neg_sids = [sid for sid in sid2sent[pid_].keys() if sid != ans_sid]

    recs = [pos_rec] if pos_rec else []
    for neg_sid in neg_sids:
        if window == 0:
            sent = sid2sent[pid_][neg_sid]
        elif window > 0:
            sent = build_sent_with_context(sid=neg_sid, pid=pid_, sid2sent=sid2sent, window=window)
        else:
            raise ValueError('Invalid window: {}'.format(window))

        rec = build_rec(yy=0, qid=qid, sid=neg_sid, question=question, sent=sent)
        recs.append(rec)

    with io.open(dump_fp, mode='a', encoding='utf-8') as dump_f:
        dump_f.writelines(recs)

    logger.info('dump: {} records for pid_: {}'.format(len(recs), pid_))


def dump_samples(data_type, window):
    headline = 'yy\tqid\tsid\tquestion\tsent\n'
    dump_fp = join(path_parser.squad_proc, 'window_{}'.format(window), '{}.tsv'.format(data_type))

    if exists(dump_fp):
        raise ValueError('dump_fp: {} exists!'.format(dump_fp))

    with io.open(dump_fp, mode='a', encoding='utf-8') as dump_f:
        dump_f.write(headline)

    qid2ans_sid, qid2question, sid2sent = build_basic_info(data_type)

    p = Pool(5)
    partial_func = partial(dump_core,
                           # qid2ans_sid=qid2ans_sid,
                           qid2question=qid2question,
                           sid2sent=sid2sent,
                           dump_fp=dump_fp,
                           window=window)

    p.map(partial_func, list(qid2ans_sid.items()))


def print_ns_distribution():
    """
        Measures the distribution over #sentences in SQuAD passages.

        Representative data points:
            2: 2.67
            4: 26.73
            8: 87.98
            10: 95.77
            16: 99.80

        max_ns is set to 16 for MIL.

    :return:
    """
    qid2ans_sid, qid2question, sid2sent = build_basic_info(data_type='train')
    ns2np = {}
    max_ns = 20
    for i in range(max_ns):
        ns2np[i] = 0

    ns = [len(sid2sent[pid]) for pid in sid2sent]

    ratios = [float(ns.count(i))/len(sid2sent) for i in range(max_ns)]
    left = 1.0 - sum(ratios)
    ratios.append(left)

    for i in range(len(ratios)):
        print('{}: {}'.format(i+1, sum(ratios[:i+1])))


def stats(data_type, window):
    dump_fp = join(path_parser.squad_proc, 'window_{}'.format(window), '{}.tsv'.format(data_type))
    with io.open(dump_fp, encoding='utf-8') as dump_f:
        lines = dump_f.readlines()

    n_pos = len([line for line in lines if line.startswith('1')])
    n_neg = len(lines) - n_pos
    logger.info('n_pos: {}, n_neg: {}. ratio: {}'.format(n_pos, n_neg, float(n_pos)/n_neg))


def validate(data_type, window):
    dump_fp = join(path_parser.squad_proc, 'window_{}'.format(window), '{}.tsv'.format(data_type))
    with io.open(dump_fp, encoding='utf-8') as dump_f:
        line = dump_f.readline()
        if len(line.split('\t')) != 5:
            raise ValueError('Invalid line: {}'.format(line))
    logger.info('Validated!')


if __name__ == '__main__':
    # data_type = 'train'
    # window = 2
    # dump_samples(data_type, window=window)
    # validate(data_type, window=window)

    print_ns_distribution()
    # stats(data_type)
