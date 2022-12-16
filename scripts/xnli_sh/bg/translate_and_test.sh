

TRAIN=my_datasets/sequence_classification/xnli/dev-bg-mini_dev.tsv
DEV=my_datasets/sequence_classification/xnli/dev-bg-mini_test.tsv
TEST=my_datasets/sequence_classification/xnli/dev-bg-mini_test.tsv


for SEED in 2 3; do
  OUT=models/xnli/jttls/bg/mBart_bg_en_pr_not_trained_seed_$SEED

  PRETRAINED_LM_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o_seed_$SEED/test/checkpoints*
  PRETRAINED_MT=models/iwslt/pretrained_mts/bg-en/bg_en_mbart_pr_model_50_m2o_max_seq_len_85/test/checkpoints_45.07BLEU

  echo "Training translate and test..."
  mkdir -p $OUT
  python -m scripts.training_sample --train_data $TRAIN --validation_data $DEV --test_data $TEST --src bg --tgt en --save_dir $OUT --batch_size 4 --gpus 1 --tokenizer $PRETRAINED_LM_EN --max_input_len 85 --model_lm_path $PRETRAINED_LM_EN --model_seq2seq_path $PRETRAINED_MT --test --seed $SEED > $OUT/log_results.txt
done
