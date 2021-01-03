# querysum
Code for *Coarse-to-Fine Query Focused Multi-Document Summarization*.

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
pip freeze > requirements.txt
pip install -r requirements.txt
```
You also need to install apex:
```bash
cd ..
git clone https://www.github.com/nvidia/apex
cd apex
python3 setup.py install
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
### Rank with IR
In `src/frame/ir/ir_tf.py`, 
```bash
rank_e2e()  # rank all sentences
ir_rank2records()  # filter sentences based on IR scores
```
Specfically, `rank_e2e()` builds a directory under `rank`, e.g., `rank/ir-tf-2007`, which stores ranking files for each cluster. On the top of it, `ir_rank2records()` builds a filtered record for each cluster, e.g., under `rank/ir_records-ir-tf-2007-0.75_ir_conf` where `0.75` is the accumulated confidence. 

You can specifiy IR configs in `ir/ir_config.py`.

### Rank with QA  (TBD)


### Rank with Centrality (TBD)

