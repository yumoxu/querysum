import logging
import logging.config
import yaml
from io import open
import os
from os.path import join, dirname, abspath
import warnings
import sys
from pytorch_transformers import BertModel, BertTokenizer, BertForQuestionAnswering
import torch
from pathlib import Path

sys.path.insert(0, dirname(dirname(abspath(__file__))))


def deprecated(func):
    """
        This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used.
    """

    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning)
        return func(*args, **kwargs)

    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


class PathParser:
    def __init__(self, proj_root):
        self.proj_root = proj_root
        self.log = join(self.proj_root, 'log')

        # set data
        self.data = join(self.proj_root, 'data')
        self.squad = join(self.data, 'squad')
        self.squad_raw = join(self.squad, 'raw')
        self.squad_proc = join(self.squad, 'proc')
        self.data_docs = join(self.data, 'docs')
        self.data_passages = join(self.data, 'passages')
        self.data_topics = join(self.data, 'topics')

        # tdqfs
        self.data_tdqfs = join(self.data, 'tdqfs')
        self.data_tdqfs_sentences = join(self.data_tdqfs, 'sentences')
        self.data_tdqfs_passages = join(self.data_tdqfs, 'passages')
        self.data_tdqfs_queries = join(self.data_tdqfs, 'query_info.txt')
        self.data_tdqfs_summary_targets = join(self.data_tdqfs, 'summary_targets')

        self.data_summary_results = join(self.data, 'summary_results')
        self.data_summary_refs = join(self.data, 'summary_refs')
        self.data_summary_targets = join(self.data, 'summary_targets')

        self.res = join(self.proj_root, 'res')
        self.model_save = join(self.proj_root, 'model')
        self.bert_qa = join(self.model_save, 'qa_sentence')

        # bert passage
        self.bert_passage_root = join(self.model_save, 'squad_passage')
        self.bert_passage_tokenizer = join(self.bert_passage_root, 'passage_tokenizer')
        self.bert_passage_checkpoint_root = join(self.bert_passage_root, 'squad-epoch_5')
        self.bert_passage_checkpoint = join(self.bert_passage_checkpoint_root, 'checkpoint-{}')
        self.bert_passage_model = join(self.bert_passage_checkpoint, 'pytorch_model.bin')
        self.bert_passage_config = join(self.bert_passage_checkpoint, 'config.json')

        self.pred = join(self.proj_root, 'pred')

        if config_meta['grain'] == 'sent':
            self.summary_rank = join(self.proj_root, 'rank')
            self.summary_text = join(self.proj_root, 'text')
            self.graph = join(self.proj_root, 'graph')
        else:
            self.summary_rank = join(self.proj_root, 'rank_{}'.format(config_meta['grain']))
            self.summary_text = join(self.proj_root, 'text_{}'.format(config_meta['grain']))
            self.graph = join(self.proj_root, 'graph_{}'.format(config_meta['grain']))
        
        self.graph_rel_scores = join(self.graph, 'rel_scores')  # for dumping relevance scores
        self.graph_token_logits = join(self.graph, 'token_logits')  # for dumping relevance scores

        self.rouge = join(self.proj_root, 'rouge')

        self.tune = join(self.proj_root, 'tune')
        self.rouge_dir = '~/ROUGE-1.5.5/data'  # specify your ROUGE dir

proj_root = os.path.dirname(os.path.dirname(__file__))
path_parser = PathParser(proj_root=proj_root)

config_root = join(proj_root, 'config')
config_meta_fp = os.path.join(config_root, 'config_meta.yml')
config_meta = yaml.load(open(config_meta_fp, 'r', encoding='utf-8'))

# model
meta_model_name = config_meta['model_name']
config_model_fn = 'config_model_{0}.yml'.format(meta_model_name)
config_model_fp = os.path.join(config_root, config_model_fn)
config_model = yaml.load(open(config_model_fp, 'r'))

test_year = config_meta['test_year']
grain = config_meta['grain']
remove_dialog = config_meta['remove_dialog']

if meta_model_name == 'bert_passage':
    from frame.bert_passage.config_name import model_name
elif meta_model_name in ('bert_qa', 'bert_base'):
    from frame.bert_qa.config_name import model_name
else:
    model_name = 'MiscModel'

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('log/{0}.log'.format(model_name))
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f'model name: {model_name}')

NARR = 'narr'
TITLE = 'title'
QUERY = 'query'
NONE = 'None'
SEP = '_'
years = ['2005', '2006', '2007']

def load_bert_qa():
    print('Load PyTorch model from {}'.format(path_parser.bert_qa))
    if not torch.cuda.is_available():  # cpu
        state = torch.load(path_parser.bert_qa, map_location='cpu')
    else:
        state = torch.load(path_parser.bert_qa)

    return state['epoch'], state['model'], state['tokenizer'], state['scores']

bert_passage_iter = 12000
def load_bert_passage():
    if meta_model_name != 'bert_passage':
        raise ValueError('Invalid meta_model_name: {}'.format(meta_model_name))

    tokenizer_dir = path_parser.bert_passage_tokenizer
    checkpoint_dir = path_parser.bert_passage_checkpoint.format(bert_passage_iter)

    print('Load PyTorch model from {}, vocab: {}'.format(checkpoint_dir, tokenizer_dir))

    model_params = {
        'pretrained_model_name_or_path': checkpoint_dir,
    }
    bert_model = BertForQuestionAnswering.from_pretrained(**model_params)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir,
                                              do_lower_case=True,
                                              do_basic_tokenize=True)

    return bert_model, tokenizer


preload_model_tokenizer = config_meta['preload_model_tokenizer']
if preload_model_tokenizer:
    if meta_model_name == 'bert_qa' and config_model['fine_tune'] == 'qa':
        logger.info('building BERT model and tokenizer: {}'.format(config_model['fine_tune']))
        _, bert_model, bert_tokenizer, _ = load_bert_qa()

    elif meta_model_name == 'bert_passage' and config_model['fine_tune'] == 'passage':
        logger.info('building BERT model and tokenizer: {}'.format(config_model['fine_tune']))
        bert_model, bert_tokenizer = load_bert_passage()

    else:
        logger.info('building BERT tokenizer')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

mode = config_meta['mode']
if mode == 'rank_sent':
    config_model['d_batch'] = 50
