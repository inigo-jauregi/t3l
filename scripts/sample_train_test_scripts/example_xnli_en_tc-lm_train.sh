#!/bin/bash

TRAIN=my_datasets/sequence_classification/xnli/train-en.tsv
DEV=my_datasets/sequence_classification/xnli/dev-en.tsv
TEST=my_datasets/sequence_classification/xnli/dev-en.tsv

OUT=models/xnli/pretrained_lms/en/mBART_xnli_english_TC-LM
PRETRAINED_EN=facebook/mbart-large-50-many-to-one-mmt

echo "Training English XNLI TC-LM model..."
mkdir -p $OUT
python -m scripts.train_lm --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT \
                           --batch_size 8 --grad_accum 2 --lr 0.000003 --gpus 1 \
                           --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN --epochs 10 \
                           --max_input_len 170 > $OUT/log_results.txt
