

TRAIN=my_datasets/sequence_classification/xnli/dev-es-mini_dev.tsv
DEV=my_datasets/sequence_classification/xnli/dev-es-mini_test.tsv
TEST=my_datasets/sequence_classification/xnli/test-es.tsv

for SEED in 1; do
  OUT=models/xnli/intermediate_translations/es/jttl/mBart_es_en_50m2o_batch1_lr3-6_seed_$SEED.samples_100

  PRETRAINED_LM_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o_seed_$SEED/test/checkpoints*
  PRETRAINED_MT=models/iwslt/pretrained_mts/es-en/es_en_mbart_pr_model_50_m2o_max_seq_len_85/test/checkpoints_52.96

  echo "Training translate and test..."
  mkdir -p $OUT
  python -m scripts.training_sample --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST --src es --tgt en --save_dir $OUT --batch_size 4 --gpus 1 --tokenizer $PRETRAINED_LM_EN --max_input_len 85 --max_output_len 85 --model_lm_path $PRETRAINED_LM_EN --model_seq2seq_path $PRETRAINED_MT --test --from_pretrained models/xnli/jttls/es/mBart_es_en_50m2o_batch1_lr3-6_seed_1.samples_100/test/checkpoints/check-epoch* --int_trans_name $OUT/int_translations.txt

done
