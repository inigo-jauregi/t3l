

TRAIN=my_datasets/sequence_classification/xnli/dev-zh-mini_dev.tsv
DEV=my_datasets/sequence_classification/xnli/dev-zh-mini_test.tsv
TEST=my_datasets/sequence_classification/xnli/dev-zh-mini_test.tsv

for SEED in 2 3; do

  OUT=models/xnli/jttls/zh/mBart_zh_en_50_m2o_no_train_seed_$SEED

  PRETRAINED_LM_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o_seed_$SEED/test/checkpoints*
  PRETRAINED_MT=models/iwslt/pretrained_mts/zh-en/zh_en_mbart_50_m2o_max_seq_len_85/test/checkpoints_19.075996931200997

  echo "Testing translate + test approach..."
  mkdir -p $OUT
  python -m scripts.training_sample --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --src zh --tgt en --batch_size 8 --gpus 1 --tokenizer $PRETRAINED_LM_EN --max_input_len 85 --max_output_len 85 --model_lm_path $PRETRAINED_LM_EN --model_seq2seq_path $PRETRAINED_MT --test --seed $SEED > $OUT/log_results.txt

done
