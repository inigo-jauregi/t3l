#!/bin/bash

TRAIN=my_datasets/sequence_classification/xnli/dev-zh-mini_dev-mini_dev_10.tsv
DEV=my_datasets/sequence_classification/xnli/dev-zh-mini_test.tsv
TEST=my_datasets/sequence_classification/xnli/test-zh.tsv  # Make sure this is the test dataset


OUT=models/xnli/t3l/zh/T3L_FS-10
PRETRAINED_LM_EN=models/xnli/pretrained_lms/en/mBART_xnli_english_TC-LM/test/checkpoints*
PRETRAINED_MT=facebook/mbart-large-50-many-to-one-mmt

echo "Training T3L few-shots..."
mkdir -p $OUT
# Note! For zero-shot T3L just remove the --from_pretrained argument
python -m scripts.training_t3l --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST \
                               --src zh --tgt en --save_dir $OUT --gpus 1 --tokenizer $PRETRAINED_LM_EN \
                               --max_input_len 85 --max_output_len 85 --model_lm_path $PRETRAINED_LM_EN \
                               --model_seq2seq_path $PRETRAINED_MT --test \
                               --from_pretrained $OUT/test/checkpoints/check-epoch* > $OUT/log_results_TEST.txt
