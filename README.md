# What's Missing: Knowledge Gap Guided QA (GapQA)

This repository contains the code used to produce the results in 
```
What's Missing: A Knowledge Gap Guided Approach for Multi-hop Question Answering
Tushar Khot, Ashish Sabharwal, Peter Clark
EMNLP 2019
```
Bibtex:
```
@inproceedings{WhatsMissing19,
  title={What's Missing: A Knowledge Gap Guided Approach for Multi-hop Question Answering},
  author={Tushar Khot and Ashish Sabharwal and Peter Clark},
  booktitle={EMNLP},
  year={2019}
}
```

Table of Contents
===============

* [Setup Environment](#setup-environment)
* [Downloading Data](#downloading-data)
    * [Dataset](#dataset)
    * [Knowledge Sources](#knowledge-sources)
    * [Trained models](#trained-models)
* [Training Models](#training-models)
* [Evaluating Models](#evaluating-models)


## Setup Environment

1. Create the `missingfact` environment using Anaconda

   ```
   conda create -n missingfact python=3.6
   ```

2. Activate the environment

   ```
   source activate missingfact
   ```

3. Install the requirements in the environment:

   ```
   pip install -r requirements.txt
   ```
4. Download NLTK stopwords

   ```
   python -m nltk.downloader stopwords
   ```

## Downloading Data

### Dataset
The Knowledge Gap Dataset (KGD) can be downloaded from [here](http://data.allenai.org/downloads/missingfact/kgd.tar.gz). To help with
running the model training experiments, we provide a script that downloads
the training datasets specifically used for the GapQA model. These training datasets have predictions
from the key term identification model as well as the retrieved sentences from our corpus.

```
./scripts/downloads/download_train_data.sh
```

These files will be saved in the `data/input/` folder.

### Knowledge Sources
Apart from the training data, GapQA also depends on ConceptNet tuples and the ARC corpora as
a source of knowledge. We provide the English subset of ConceptNet tuples used by our model
along with WordNet+OMCS subset used in our ablations.
```
./scripts/downloads/download_cn.sh
```
These files will be saved in the `data/conceptnet/` folder.

Since the ARC corpora is extremely large and requires an ElasticSearch instance, we provide
JSONL files containing prefetched sentences in the training datasets above. If you would
like to change the retrieval, the ARC corpus can be downloaded
[here](http://data.allenai.org/arc/arc-corpus/)

### Trained models
We provide all the trained GapQA models that were reported in our experiments. You can
download them using:
```
./scripts/downloads/download_models.sh
```
The models will be downloaded to `data/trained_models/`.
 
## Training Models
  If you want to re-train the GapQA models instead, we provide a helper script to train N models
  for a given AllenNLP training config.
  ```
  NUM=5 \
  OUTDIR=trained_models/fullmodel \
  CONF=training_configs/full_model_shortans.json \
  ./scripts/train_n_models.sh
  ```
  NUM specifies the number of models trained, OUTDIR is the output location of the saved models,
  and CONF is the training config.

  For each of our experiment, here are the commands that you can run to train the corresponding models.
  These configs assume the files are downloaded to the appropriate location using the
  `download_train_data.sh` script. The configs file make sure to load the right training files
  based on the experiment setting. All the trained models will be saved to `trained_models/`.

### Full Model
```
NUM=5 \
OUTDIR=trained_models/fullmodel \
CONF=training_configs/full_model_shortans.json \
./scripts/train_n_models.sh
```

### Full Model (f  + OMCS)
```
NUM=5 \
OUTDIR=trained_models/ablation_model_omcs \
CONF=training_configs/ablation_model_omcs.json \
./scripts/train_n_models.sh
```

### Full Model (f  + WordNet)
```
NUM=5 \
OUTDIR=trained_models/ablation_model_wordnet \
CONF=training_configs/ablation_model_wordnet.json \
./scripts/train_n_models.sh
```

### Ablations: No Annotations
```
NUM=5 \
OUTDIR=trained_models/ablation_model_noann \
CONF=training_configs/ablation_model_noann_shortans.json \
./scripts/train_n_models.sh
```

### Ablations: Heur. Span Anns
```
NUM=5 \
OUTDIR=trained_models/ablation_model_heurspans \
CONF=training_configs/ablation_model_heurspans_shortans.json \
./scripts/train_n_models.sh
```

### Ablations: No Relation Score
```
NUM=5 \
OUTDIR=trained_models/ablation_model_norelscore \
CONF=training_configs/ablation_model_norelscore_shortans.json \
./scripts/train_n_models.sh
```

### Ablations: No Spans (Model)
```
NUM=5 \
OUTDIR=trained_models/ablation_model_nospanmod \
CONF=training_configs/ablation_model_nospanmod_shortans.json \
./scripts/train_n_models.sh
```

### Ablations: No Spans (IR)
```
NUM=5 \
OUTDIR=trained_models/ablation_model_nospanir \
CONF=training_configs/ablation_model_nospanir_shortans.json \
./scripts/train_n_models.sh
```

## Evaluating Models
Similarly, we provide a helper script to evaluate the N trained models against the corresponding
evaluation files. Note that depending on the particular ablation experiment, the evaluations
will be run against a different file. E.g. the ablation experiment that does not use spans for
IR should be evaluated against the test file where the prefetched sentences ignore the spans too.

For each of our experiment, here are the commands that you can run to evaluate the downloaded
models (using the `download_models.sh` script). To use your trained models, just replace the
`data/trained_models` directory with  `trained_models/`. The accuracies will be reported in
`accuracies_...jsonl` file within the input model directory.

### Full Model
```
NUM=5 \
OUTDIR=data/trained_models/fullmodel \
EVAL=data/input/shortans/test_prefetched_SPAN_PRED.jsonl \
./scripts/eval_n_model.sh
```

### Full Model (f  + OMCS)
```
NUM=5 \
OUTDIR=data/trained_models/ablation_model_omcs \
EVAL=data/input/shortans/test_prefetched_SPAN_PRED.jsonl \
./scripts/eval_n_model.sh
```

### Full Model (f  + WordNet)
```
NUM=5 \
OUTDIR=trained_models/ablation_model_wordnet \
EVAL=data/input/shortans/test_prefetched_SPAN_PRED.jsonl \
./scripts/eval_n_model.sh
```

### Ablations: No Annotations
```
NUM=5 \
OUTDIR=trained_models/ablation_model_noann \
EVAL=data/input/shortans_squadspans/test_prefetched_SPAN_PRED.jsonl \
./scripts/eval_n_model.sh
```

### Ablations: Heur. Span Anns
```
NUM=5 \
OUTDIR=trained_models/ablation_model_heurspans \
EVAL=data/input/shortans_heurspans/test_prefetched_SPAN_PRED.jsonl \
./scripts/eval_n_model.sh
```

### Ablations: No Relation Score
```
NUM=5 \
OUTDIR=trained_models/ablation_model_norelscore \
EVAL=data/input/shortans/test_prefetched_SPAN_PRED.jsonl \
./scripts/eval_n_model.sh
```

### Ablations: No Spans (Model)
```
NUM=5 \
OUTDIR=trained_models/ablation_model_nospanmod \
EVAL=data/input/shortans/test_prefetched_SPAN_PRED.jsonl \
./scripts/eval_n_model.sh
```

### Ablations: No Spans (IR)
```
NUM=5 \
OUTDIR=trained_models/ablation_model_nospanir \
EVAL=data/input/shortans_nospanir/test_prefetched_SPAN_PRED.jsonl \
./scripts/eval_n_model.sh
```
