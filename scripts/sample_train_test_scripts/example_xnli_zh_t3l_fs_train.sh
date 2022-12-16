#!/bin/bash

TRAIN=my_datasets/sequence_classification/xnli/dev-zh-mini_dev-mini_dev_10.tsv
DEV=my_datasets/sequence_classification/xnli/dev-zh-mini_test.tsv
TEST=my_datasets/sequence_classification/xnli/dev-zh-mini_test.tsv


OUT=models/xnli/t3l/zh/T3L_FS-10_seed_1234
PRETRAINED_LM_EN=models/xnli/pretrained_lms/en/mBART_xnli_english_TC-LM/test/checkpoints*
PRETRAINED_MT=facebook/mbart-large-50-many-to-one-mmt

echo "Training T3L few-shots..."
mkdir -p $OUT
python -m scripts.training_t3l --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST \
                               --src zh --tgt en --save_dir $OUT --batch_size 1 --freeze_strategy fix_nmt_dec_lm_enc \
                               --gpus 1 --tokenizer $PRETRAINED_LM_EN --max_input_len 85 --max_output_len 85 \
                               --warmup 0 --lr 0.000003 --model_lm_path $PRETRAINED_LM_EN \
                               --model_seq2seq_path $PRETRAINED_MT --epochs 10 --seed 1234 > $OUT/log_results.txt
