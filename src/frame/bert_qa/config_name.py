import yaml
from io import open
import os
from os.path import join, dirname, abspath
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))

config_root = join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')

# meta
config_meta_fp = os.path.join(config_root, 'config_meta.yml')
config_meta = yaml.load(open(config_meta_fp, 'r', encoding='utf-8'))

# model
meta_model_name = config_meta['model_name']
config_model_fn = 'config_model_{0}.yml'.format(meta_model_name)
config_model_fp = os.path.join(config_root, config_model_fn)
config_model = yaml.load(open(config_model_fp, 'r'))

if config_model['model_name']:
    model_name = config_model['model_name']
