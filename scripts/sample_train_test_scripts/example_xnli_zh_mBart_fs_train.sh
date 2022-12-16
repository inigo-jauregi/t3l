#!/bin/bash

TRAIN=my_datasets/sequence_classification/xnli/dev-zh-mini_dev-mini_dev_10.tsv
DEV=my_datasets/sequence_classification/xnli/dev-zh-mini_test.tsv
TEST=my_datasets/sequence_classification/xnli/dev-zh-mini_test.tsv

OUT=models/xnli/mBART/zh/mBART_FS-10_seed_1234
PRETRAINED_EN=models/xnli/pretrained_lms/en/mBART_xnli_english_TC-LM/test/checkpoints*

echo "Training mBART few shot model..."
mkdir -p $OUT
python -m scripts.train_lm --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST \
                           --save_dir $OUT --batch_size 8 --grad_accum 2 --lr 0.000003 --gpus 1 \
                           --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN --epochs 10 \
                           --seed 1234 --max_input_len 170 > $OUT/log_results.txt
