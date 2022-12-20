#!/bin/bash

TRAIN=my_datasets/translation/IWSLT_2014_TEDtalks/el-en/el-en/train
DEV=my_datasets/translation/IWSLT_2014_TEDtalks/el-en/el-en/dev
TEST=my_datasets/translation/IWSLT_2014_TEDtalks/el-en/el-en/test

SRC=el
TGT=en

OUT=models/iwslt/pretrained_mts/el-en/MT-LM_IWSLT_tuning
PRETRAINED_LM=facebook/mbart-large-50-many-to-one-mmt

echo "Training translator..."
mkdir -p $OUT
python -m scripts.train_translation --train_data $TRAIN --validation_data $DEV --test_data $TEST \
                                    --src $SRC --tgt $TGT --save_dir $OUT --tokenizer $PRETRAINED_LM \
                                    --model_lm_path $PRETRAINED_LM --batch_size 8 --grad_accum 2 \
                                    --gpus 1 --epochs 10 --max_input_len 170 > $OUT/log_results.txt
