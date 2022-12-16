# T3L: Translate and test transfer learning for cross-lingual text classification

The code reporitory for the paper research paper with the same title.

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