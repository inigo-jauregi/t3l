# T3L: Translate-and-test transfer learning for cross-lingual text classification

The code reporitory for the paper research [paper published in the Transactions of the Association for Computational Linguistics (TACL)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00593/117584/T3L-Translate-and-Test-Transfer-Learning-for-Cross).

## Installation

Python version > 3.6.

Install required packages with pip:

python
```
pip install -r requirements.txt

# Install our adapted transformers package
cd t3l/transformers
pip install -e .
cd ../..
```

## Data

The XNLI and MultiEurlex datasets and the respective few-shot 10 and few-shot 100
data splits, as well as the TED talk translation corpus used in the paper
for the fine-tuning of the MT-LM models of bg-en, el-en and sw-en can be 
found [here](https://drive.google.com/drive/folders/1ZJTViKBQ4B2PO9OA04PjsT1m1JtEkYLK?usp=sharing).

Downloading MLdoc data requires an approval from 
[TREC](https://trec.nist.gov/data/reuters/reuters.html) and to follow
the [script](https://github.com/facebookresearch/MLDoc) provided by 
the MLdoc paper authors to extract the benchmark dataset.

## Getting started

The following commands show how to fine-tune the different
described in the paper.

### Fine-tune MT-LM
```bash
python -m scripts.train_translation --train_data $TRAIN \
                                    --validation_data $DEV \
                                    --test_data $TEST \
                                    --src $SRC \
                                    --tgt $TGT \
                                    --save_dir $OUT \
                                    --tokenizer $PRETRAINED_LM \
                                    --model_lm_path $PRETRAINED_LM \
                                    --batch_size $BATCH_SIZE \
                                    --grad_accum $GRAD_ACCUM \
                                    --gpus $NUM_GPUS \
                                    --epochs $EPOCHS
```

### Fine-tune TC-LM

```bash
python -m scripts.train_lm --task XNLI \ 
                           --train_data $TRAIN \
                           --validation_data $DEV \
                           --test_data $TEST \
                           --save_dir $OUT \
                           --batch_size $BATCH_SIZE \
                           --grad_accum $GRAD_ACCUM \
                           --lr $LEARNING_RATE \
                           --gpus $NUM_GPUS \
                           --tokenizer $PRETRAINED_EN \
                           --model_lm_path $PRETRAINED_EN \
                           --epochs $EPOCH
```


### T3L
```bash
python -m scripts.training_t3l --task XNLI \
                               --train_data $TRAIN \
                               --validation_data $DEV \
                               --test_data $TEST \
                               --src $SRC \
                               --tgt $TGT \
                               --save_dir $OUT \
                               --batch_size $BATCH_SIZE \
                               --grad_accum $GRAD_ACCUM \
                               --freeze_strategy fix_nmt_dec_lm_enc \
                               --gpus $NUM_GPUS \
                               --tokenizer $PRETRAINED_LM_EN \
                               --max_input_len $MAX_SEQ_LEN_INPUT \
                               --max_output_len $MAX_SEQ_LEN_OUTPUT \
                               --warmup $WARMUP \
                               --lr $LEARNING_RATE \
                               --model_lm_path $PRETRAINED_LM_EN \
                               --model_seq2seq_path $PRETRAINED_MT \
                               --epochs $EPOCHS

```

### Testing models

To perform inference (test) over the test sets with the trained
models, simply add the `--test` flag and the `--from_pretrained` argument
providing the path to the trained checkpoint.

## Citation

Please cite our paper in your work:

```latex
@article{10.1162/tacl_a_00593,
    author = {Jauregi Unanue, Inigo and Haffari, Gholamreza and Piccardi, Massimo},
    title = "{T3L: Translate-and-Test Transfer Learning for Cross-Lingual Text Classification}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {11},
    pages = {1147-1161},
    year = {2023},
    month = {09},
    abstract = "{Cross-lingual text classification leverages text classifiers trained in a high-resource language to perform text classification in other languages with no or minimal fine-tuning (zero/ few-shots cross-lingual transfer). Nowadays, cross-lingual text classifiers are typically built on large-scale, multilingual language models (LMs) pretrained on a variety of languages of interest. However, the performance of these models varies significantly across languages and classification tasks, suggesting that the superposition of the language modelling and classification tasks is not always effective. For this reason, in this paper we propose revisiting the classic “translate-and-test” pipeline to neatly separate the translation and classification stages. The proposed approach couples 1) a neural machine translator translating from the targeted language to a high-resource language, with 2) a text classifier trained in the high-resource language, but the neural machine translator generates “soft” translations to permit end-to-end backpropagation during fine-tuning of the pipeline. Extensive experiments have been carried out over three cross-lingual text classification datasets (XNLI, MLDoc, and MultiEURLEX), with the results showing that the proposed approach has significantly improved performance over a competitive baseline.}",
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00593},
    url = {https://doi.org/10.1162/tacl\_a\_00593},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00593/2159097/tacl\_a\_00593.pdf},
}
```
