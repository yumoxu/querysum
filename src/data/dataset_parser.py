# -*- coding: utf-8 -*-
import io
from os.path import dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

import re
from os import listdir
from os.path import join
import itertools

import utils.config_loader as config
from utils.config_loader import logger, path_parser, config_meta, config_model
import utils.tools as tools

import data.clip_and_mask_sl as cm_sl

import nltk
from nltk.tokenize import sent_tokenize, TextTilingTokenizer

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.porter import PorterStemmer

"""

    This class provide following information extraction functions:

    - get_doc:
        Get an article from file.
        Return natural paragraphs/subtopic tiles/whole artilce.

    - doc2sents:
        Based on func:get_doc, get sentences from an article.
        Optionally, sentences can be organized by paragraphs.

    - cid2sents:
        Based on func:doc2sents, get sentences from a cluster.

    This class also provide following parsing functions:

    - parse_doc2sents:
        Based on func:doc2sents,

"""

class DatasetParser:
    def __init__(self):
        # info
        self.cluster_info = dict()
        self.article_info = dict()

        if config_meta['word_tokenizer'] == 'bert':
            self.word_tokenize = config.bert_tokenizer.tokenize
        elif config_meta['word_tokenizer'] == 'nltk':
            self.word_tokenize = nltk.tokenize.word_tokenize
        else:
            raise ValueError('Invalid word_tokenizer: {}'.format(config_meta['word_tokenizer']))

        self.sent_tokenize = nltk.tokenize.sent_tokenize
        self.porter_stemmer = PorterStemmer()

        if config_meta['texttiling']:
            self.para_tokenize = TextTilingTokenizer()

        # base pat
        BASE_PAT = '(?<=<{0}> )[\s\S]*?(?= </{0}>)'
        BASE_PAT_WITH_NEW_LINE = '(?<=<{0}>\n)[\s\S]*?(?=\n</{0}>)'
        BASE_PAT_WITH_RIGHT_NEW_LINE = '(?<=<{0}>)[\s\S]*?(?=\n</{0}>)'

        # query pat
        self.id_pat = re.compile(BASE_PAT.format('num'))
        self.title_pat = re.compile(BASE_PAT.format('title'))
        self.narr_pat = re.compile(BASE_PAT_WITH_NEW_LINE.format('narr'))

        # article pat
        self.text_pat = re.compile(BASE_PAT_WITH_NEW_LINE.format('TEXT'))
        self.graphic_pat = re.compile(BASE_PAT_WITH_NEW_LINE.format('GRAPHIC'))
        self.type_pat = re.compile(BASE_PAT_WITH_NEW_LINE.format('TYPE'))
        self.para_pat = re.compile(BASE_PAT_WITH_NEW_LINE.format('P'))

        self.proc_params_for_questions = {
            'rm_dialog': False,
            'rm_stop': False,
            'stem': True,
        }

    def _get_word_ids(self, words):
        word_ids = config.bert_tokenizer.convert_tokens_to_ids(words)
        return word_ids

    def _proc_sent(self, sent, rm_dialog, rm_stop, stem, rm_short=None, min_nw_sent=3):
        sent = sent.lower()
        sent = re.sub(r'\s+', ' ', sent).strip()  # remove extra spaces

        if not sent:
            return None

        if rm_short and len(nltk.tokenize.word_tokenize(sent)) < min_nw_sent:
            return None

        if rm_dialog:
            dialog_tokens = ["''", "``"]
            for tk in dialog_tokens:
                if tk in sent:
                    logger.info('Remove dialog')
                    return None

            if config.test_year == '2005' and sent[0] == "'" and ('says' in sent or 'said' in sent):
                logger.info('Remove dialog')
                return None

        if rm_stop:
            sent = remove_stopwords(sent)

        if stem:
            sent = self.porter_stemmer.stem_sentence(sent)

        return sent

    def _proc_para(self, pp, rm_dialog=True, rm_stop=True, stem=True, to_str=False):
        """
            Return both original paragraph and processed paragraph.

        :param pp:
        :param rm_dialog:
        :param rm_stop:
        :param stem:
        :param to_str: if True, concatenate sentences and return.
        :return:
        """
        original_para_sents, processed_para_sents = [], []

        for ss in self.sent_tokenize(pp):
            ss_origin = self._proc_sent(ss, rm_dialog=False, rm_stop=False, stem=False)
            ss_proc = self._proc_sent(ss, rm_dialog=rm_dialog, rm_stop=rm_stop, stem=stem)

            if ss_proc:  # make sure the sent is not removed, i.e., is not empty and is not in a dialog
                original_para_sents.append(ss_origin)
                processed_para_sents.append(ss_proc)

        if not to_str:
            return original_para_sents, processed_para_sents

        para_origin = ' '.join(original_para_sents)
        para_proc = ' '.join(processed_para_sents)
        return para_origin, para_proc

    def get_doc(self, fp, concat_paras):
        """
            get an article from file.

            first get all natural paragraphs in the text, then:
                if concat_paras, return paragraphs joint by \n;
                if using texttiling, return subtopic tiles;
                if not above, return paragraphs.

        """
        with io.open(fp, encoding='utf-8') as f:
            article = f.read()

        pats = [self.text_pat, self.graphic_pat]

        PARA_SEP = '\n\n'

        for pat in pats:
            text = re.search(pat, article)

            if not text:
                continue

            text = text.group()

            # if there is '<p>' in text, gather them to text
            paras = re.findall(self.para_pat, text)
            if paras:
                text = PARA_SEP.join(paras)

            if concat_paras:
                return text

            # for text tiling: if paragraph break is a single '\n', double it
            pattern = re.compile("[ \t\r\f\v]*\n[ \t\r\f\v]*\n[ \t\r\f\v]*")
            matches = pattern.finditer(text)
            if not matches:
                text.replace('\n', PARA_SEP)

            if config_meta['texttiling']:
                try:
                    tiles = self.para_tokenize.tokenize(text)
                except ValueError:  # return short text a tiles
                    tiles = [text]

                return tiles

            if paras:
                return paras
            else:
                return text.split(PARA_SEP)

        logger.warning('No article content in {0}'.format(fp))
        return None

    def doc2sents(self, fp, para_org=False, rm_dialog=True, rm_stop=True, stem=True, rm_short=None):
        """
        :param fp:
        :param para_org: bool

        :return:
            if para_org=True, 2-layer nested lists;
            else: flat lists.

        """
        paras = self.get_doc(fp, concat_paras=False)

        original_sents, processed_sents = [], []

        if not paras:
            return [], []

        for pp in paras:
            original_para_sents, processed_para_sents = self._proc_para(pp, rm_dialog=rm_dialog,
                                                                        rm_stop=rm_stop, stem=stem)

            if para_org:
                original_sents.append(original_para_sents)
                processed_sents.append(processed_para_sents)
            else:
                original_sents.extend(original_para_sents)
                processed_sents.extend(processed_para_sents)

        return original_sents, processed_sents

    def doc2paras(self, fp, rm_dialog=True, rm_stop=True, stem=True):
        paras = self.get_doc(fp, concat_paras=False)

        if not paras:
            return [], []

        original_paras, processed_paras = [], []
        for pp in paras:
            original_para_sents, processed_para_sents = self._proc_para(pp, rm_dialog=rm_dialog,
                                                                        rm_stop=rm_stop, stem=stem)

            para_origin = ' '.join(original_para_sents)
            para_proc = ' '.join(processed_para_sents)

            original_paras.append(para_origin)
            processed_paras.append(para_proc)

        return original_paras, processed_paras

    def cid2sents(self, cid, rm_dialog=True, rm_stop=True, stem=True, max_ns_doc=None):
        """
            Load all sentences in a cluster.

        :param cid:
        :param rm_dialog:
        :param rm_stop:
        :param stem:
        :param max_ns_doc:
        :return: a 2D list.
        """

        original_sents, processed_sents = [], []
        doc_ids = tools.get_doc_ids(cid, remove_illegal=rm_dialog)  # if rm dialog, rm illegal docs.
        for did in doc_ids:
            doc_fp = tools.get_doc_fp(did)

            # 2d if para_org==True; 1d otherwise.
            original_doc_sents, processed_doc_sents = dataset_parser.doc2sents(fp=doc_fp,
                                                                               para_org=config_meta['para_org'],
                                                                               rm_dialog=rm_dialog,
                                                                               rm_stop=rm_stop,
                                                                               stem=stem)

            if max_ns_doc:
                original_doc_sents = original_doc_sents[:max_ns_doc]
                processed_doc_sents = processed_doc_sents[:max_ns_doc]

            original_sents.append(original_doc_sents)
            processed_sents.append(processed_doc_sents)

        return original_sents, processed_sents

    def cid2sents_tdqfs(self, cid):
        cc_dp = join(path_parser.data_tdqfs_sentences, cid)
        fns = [fn for fn in listdir(cc_dp)]
        original_sents, processed_sents = [], []
        for fn in fns:
            sentences = [ss.strip('\n') for ss in io.open(join(cc_dp, fn)).readlines()]
            original_doc_sents, processed_doc_sents = [], []
            for ss in sentences:
                ss_origin = self._proc_sent(ss, rm_dialog=False, rm_stop=False, stem=False)
                ss_proc = self._proc_sent(ss, rm_dialog=False, rm_stop=True, stem=True)

                if ss_proc:
                    original_doc_sents.append(ss_origin)
                    processed_doc_sents.append(ss_proc)
            
            original_sents.append(original_doc_sents)
            processed_sents.append(processed_doc_sents)

        return original_sents, processed_sents
    
    
    def cid2paras(self, cid, rm_dialog=True, rm_stop=True, stem=True, max_np_doc=None):
        original_paras, processed_paras = [], []
        doc_ids = tools.get_doc_ids(cid, remove_illegal=rm_dialog)  # if rm dialog, rm illegal docs.
        for did in doc_ids:
            doc_fp = tools.get_doc_fp(did)
            original_doc_paras, processed_doc_paras = dataset_parser.doc2paras(fp=doc_fp,
                                                                               rm_dialog=rm_dialog,
                                                                               rm_stop=rm_stop,
                                                                               stem=stem)

            if max_np_doc:
                original_doc_paras = original_doc_paras[:max_np_doc]
                processed_doc_paras = processed_doc_paras[:max_np_doc]

            original_paras.append(original_doc_paras)
            processed_paras.append(processed_doc_paras)

        return original_paras, processed_paras

    def parse(self, para, clip_and_mask, offset, rm_dialog, rm_stop, stem):
        """
            parse a para and organize results by words.
        """
        sents = [self.word_tokenize(self._proc_sent(sent, rm_dialog, rm_stop, stem))
                 for sent in self.sent_tokenize(para)]

        if not clip_and_mask:  # you only index after clipped
            return sents

        return clip_and_mask(sents, offset, join_query_para=config.join_query_para)

    def parse_query(self, query):
        """
            parse a query string => a dict with keys: ('words', 'sent_mask').
        """
        if 'max_nw_query' not in config_model:
            raise ValueError('Specify max_nw_query in config to clip query!')

        return self.word_tokenize(query)[:config_model['max_nw_query']]

    def parse_doc2sents(self, fp):
        """
            From file, parse a doc => a dict with keys:
                {'sents', 'doc_masks'}.

            The value of 'sents':
                2D nested list; each list consists of (clipped) sentence words.
        """
        _, processed_sents = self.doc2sents(fp, para_org=False, rm_dialog=True, rm_stop=False, stem=False)

        if not processed_sents:
            return None

        sents = [self.sent2words(sent) for sent in processed_sents]
        res = cm_sl.clip_and_mask_doc_sents(sents=sents)
        return res

    def sent2words(self, sent):
        """
            tokenize the given proprocessed sent.

        :param sent:
        :return:
        """
        return self.word_tokenize(sent)

    def parse_rel_sents_file(self, rel_sents_fp, rm_dialog=True, rm_stop=True, stem=True):
        """

        """
        original_sents, processed_sents = [], []

        with io.open(rel_sents_fp, encoding='utf-8', mode='r') as relv_sents_f:
            sents = [sent.strip('\n') for sent in relv_sents_f.readlines()]

        for sent in sents:
            ss_proc = self._proc_sent(sent, rm_dialog=rm_dialog, rm_stop=rm_stop, stem=stem)

            if ss_proc:  # make sure the sent is not removed, i.e., is not empty and is not in a dialog
                original_sents.append(sent)
                processed_sents.append(ss_proc)

        return original_sents, processed_sents

    def parse_summary(self, fp):
        sent_as_line = fp.split('/')[-2] != '2007'
        with io.open(fp, encoding='latin1') as f:
            content = f.readlines()

        lines = [ll.rstrip('\n') for ll in content]

        if sent_as_line:
            return lines

        sents = list(itertools.chain(*[self.sent_tokenize(ll) for ll in lines]))
        return sents

    def build_query_info(self, year, tokenize_narr, concat_title_narr=False, proc=True):
        fp = join(path_parser.data_topics, '{}.sgml'.format(year))
        with io.open(fp, encoding='utf-8') as f:
            article = f.read()
        segs = article.split('\n\n\n')
        query_info = dict()
        for seg in segs:
            seg = seg.rstrip('\n')
            if not seg:
                continue
            query_id = re.search(self.id_pat, seg)
            title = re.search(self.title_pat, seg)
            narr = re.search(self.narr_pat, seg)

            if not query_id:
                logger.info('no query id in {0} in {1}...'.format(seg, year))
                break

            if not title:
                raise ValueError('no title in {0}...'.format(seg))
            if not narr:
                raise ValueError('no narr in {0}...'.format(seg))

            query_id = query_id.group()
            title = title.group()
            narr = narr.group()  # containing multiple sentences

            if proc:
                title = self._proc_sent(sent=title, rm_dialog=False, rm_stop=False, stem=True)

            if not title:
                raise ValueError('no title in {0}...'.format(seg))

            if tokenize_narr:
                narr = sent_tokenize(narr)
                if type(narr) != list:
                    narr = [narr]

                if proc:
                    narr = [self._proc_sent(sent=narr_sent, **self.proc_params_for_questions)
                            for narr_sent in narr]
            elif proc:
                    narr = self._proc_sent(sent=narr, **self.proc_params_for_questions)

            if not narr:
                raise ValueError('no narr in {0}...'.format(seg))

            cid = config.SEP.join((year, query_id))

            if not concat_title_narr:
                query_info[cid] = {config.TITLE: title,
                                   config.NARR: narr,  # str or list
                                   }
                continue

            # concat title and narr
            if tokenize_narr:  # list
                narr.insert(0, title)  # narr is a list
                query_info[cid] = narr
            else:  # str
                sep = '. '
                if title.endswith('.'):
                    sep = sep[-1]
                title = 'describe ' + title
                query_info[cid] = sep.join((title, narr))

        return query_info

    def get_cid2query(self, tokenize_narr):
        query_dict = dict()
        for year in config.years:
            annual_dict = self.build_query_info(year, tokenize_narr, concat_title_narr=True)
            query_dict = {
                **annual_dict,
                **query_dict,
            }
        return query_dict

    def get_cid2title(self):
        title_dict = dict()
        for year in config.years:
            annual_dict = self.build_query_info(year, tokenize_narr=False, concat_title_narr=False)
            for cid in annual_dict:
                annual_dict[cid] = annual_dict[cid][config.TITLE]

            title_dict = {
                **annual_dict,
                **title_dict,
            }
        return title_dict

    def get_cid2narr(self):
        title_dict = dict()
        for year in config.years:
            annual_dict = self.build_query_info(year, tokenize_narr=False, concat_title_narr=False)
            for cid in annual_dict:
                annual_dict[cid] = annual_dict[cid][config.NARR]

            title_dict = {
                **annual_dict,
                **title_dict,
            }
        return title_dict


dataset_parser = DatasetParser()
