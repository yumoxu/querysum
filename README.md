# querysum


This repository releases the code for Coarse-to-Fine Query Focused Multi-Document Summarization. 
Please cite the following paper [[bib]](https://www.aclweb.org/anthology/2020.emnlp-main.296.bib) if you use this code,

Xu, Yumo, and Mirella Lapata. "Coarse-to-Fine Query Focused Multi-Document Summarization." In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 3632-3645. 2020.

> We consider the problem of better modeling query-cluster interactions to facilitate query focused multi-document summarization. Due to the lack of training data, existing work relies heavily on retrieval-style methods for assembling query relevant summaries. We propose a coarse-to-fine modeling framework which employs progressively more accurate modules for estimating whether text segments are relevant, likely to contain an answer, and central. The modules can be independently developed and leverage training data if available. We present an instantiation of this framework with a trained evidence estimator which relies on distant supervision from question answering (where various resources exist) to identify segments which are likely to answer the query and should be included in the summary. Our framework is robust across domains and query types (i.e., long vs short) and outperforms strong comparison systems on benchmark datasets.

Should you have any query please contact me at yumo.xu@ed.ac.uk.

## Project structure
```bash
querysum
└───requirements.txt
└───README.md
└───log  # logging files
└───src  # source files
└───data  # test sets
└───graph  # graph components for centrality estimation, e.g., sim matrix and relevance vector
└───model  # QA models for infernece
└───rank  # ranking lists from sentence-level model
└───text  # predicted summaries from sentence-level model
└───rank_passage  # ranking lists from passage-level model
└───text_passage  # predicted summaries from passage-level model
```

After cloning this project, use the following command to initialize the structure:
```bash
mkdir log data graph model rank text rank_passage text_passage
```


## Create environment
```bash
cd ..
virtualenv -p python3.6 querysum
cd querysum
. bin/activate
pip install -r requirements.txt
```
You need to install apex:
```bash
cd ..
git clone https://www.github.com/nvidia/apex
cd apex
python3 setup.py install
```

Also, you need to setup ROUGE evaluation if you have not yet done it. Please refer to [this](https://github.com/bheinzerling/pyrouge) repository. After finishing the setup, specify the ROUGE path in `frame/utils/config_loader.py` as an attribute of `PathParser`:
```python
self.rouge_dir = '~/ROUGE-1.5.5/data'  # specify your ROUGE dir
```

## Prepare benchmark data
since we are not allowed to distribute DUC data, you can request DUC 2005-2007 from [NIST](https://www-nlpir.nist.gov/projects/duc/data.html).

TD-QFS data can be downloaded [here](https://www.cs.bgu.ac.il/~talbau/TD-QFS/files/TD-QFS.zip).
You can also use the processed version [here]().

After data preparation, you should have the following directory structure with the right files under each folder:
```bash
querysum
└───data
│   └───docs  # DUC clusters 
│   └───passages  # DUC passage objects for passage-level QA 
│   └───topics  # DUC queries
│   └───summary_targets  # DUC reference summaries
│   └───tdqfs  # documents, queries and reference summaries in TD-QFS
```


## Run experiments
We go though the three stages in the QuerySum pipeline: retrieval, answering, and summarization.

### Retrieval
In `src/frame/ir/ir_tf.py`, 
```python
rank_e2e()  # rank all sentences
ir_rank2records()  # filter sentences based on IR scores
```
Specfically, `rank_e2e` builds a directory under `rank`, e.g., `rank/ir-tf-2007`, which stores ranking files for each cluster. On the top of it, `ir_rank2records` builds a filtered record for each cluster, e.g., under `rank/ir_records-ir-tf-2007-0.75_ir_conf` where `0.75` is the accumulated confidence. 

You can specifiy IR configs in `src/frame/ir/ir_config.py`.

### Answering
For sentence-level QA (i.e., answer sentence selection), in `src/frame/bert_qa/main.py`,
```python
run(prep=True, mode='rec')
```
Note that you only need to set `prep=True` at the first time, which calculates and saves query relevance scores for the retrieved sentences from the last module. 
The scores are then converted into a ranking list, from which top K sentences are selected.

You can specify QA config in `src/frame/bert_qa/qa_config.py`.

For passage-level QA (i.e., MRC), use `src/frame/bert_passage/infer.py`.

Trained models used in the paper can be downloaded [here](https://drive.google.com/file/d/1lOb9ECZa_fsYCI7Q41xMQjL0fzFvpkkD/view?usp=sharing).

### Summarization
In `src/frame/centrality/centrality_qa_tfidf_hard.py`, run the following methods in order (or at once):

```python
build_components_e2e()  # build graph compnents
score_e2e()  # run Markov Chain
rank_e2e()  # rank sentences
select_e2e()  # compose summary
```
Specifically, `build_components_e2e` builds similarity matrix and query relevance vector for the selected sentences from the last step. `score_e2e` runs a query focused Markov Chain till convergence.
`rank_e2e` ranks sentences considering both the saliance (stationary distribution) and redundancy. Finally, `select_e2e` composes summary from the ranking.

You can specifiy centrality configs in `src/frame/ir/centrality_config.py`.
