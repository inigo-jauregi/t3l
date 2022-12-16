#!/bin/bash

TRAIN=my_datasets/sequence_classification/xnli/dev-zh-mini_dev-mini_dev_10.tsv
DEV=my_datasets/sequence_classification/xnli/dev-zh-mini_test.tsv
TEST=my_datasets/sequence_classification/xnli/test-zh.tsv  # Make sure this is the test dataset

OUT=models/xnli/few_shot/el/mBART_FS-10_seed_1234
PRETRAINED_EN=models/xnli/few_shot/el/mBART_FS-10_seed_1234/test/checkpoints*  # Only keep best checkpoint

echo "Training mBART few shot model..."
mkdir -p $OUT
python -m scripts.train_lm --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST \
                           --save_dir $OUT --gpus 1 --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN \
                           --max_input_len 170 --test > $OUT/log_results_TEST.txt
