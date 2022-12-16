
for SAMPLES in 10 100; do
  TRAIN=my_datasets/sequence_classification/xnli/dev-es-mini_dev-mini_dev_$SAMPLES.tsv
  DEV=my_datasets/sequence_classification/xnli/dev-es-mini_test.tsv
  TEST=my_datasets/sequence_classification/xnli/dev-es-mini_test.tsv

  OUT=models/xnli/jttls/es/mBart_es_en_50m2o_batch1_lr3-6_seed_1.samples_$SAMPLES

  PRETRAINED_LM_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o_seed_1/test/checkpoints*
  PRETRAINED_MT=models/iwslt/pretrained_mts/es-en/es_en_mbart_pr_model_50_m2o_max_seq_len_85/test/checkpoints_52.96

  echo "Training JTTL es..."
  mkdir -p $OUT
  python -m scripts.training_sample --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST --src es --tgt en --save_dir $OUT --batch_size 1 --freeze_strategy fix_nmt_dec_lm_enc --gpus 1 --tokenizer $PRETRAINED_LM_EN --max_input_len 85 --max_output_len 85 --warmup 0 --lr 0.000003 --model_lm_path $PRETRAINED_LM_EN --model_seq2seq_path $PRETRAINED_MT --epochs 10 --seed 1234 > $OUT/log_results.txt 

  for SEED in 2 3; do

    OUT=models/xnli/jttls/es/mBart_es_en_50m2o_batch1_lr3-6_seed_$SEED.samples_$SAMPLES

    PRETRAINED_LM_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o_seed_$SEED/test/checkpoints*
    PRETRAINED_MT=models/iwslt/pretrained_mts/es-en/es_en_mbart_pr_model_50_m2o_max_seq_len_85/test/checkpoints_52.96

    echo "Training JTTL es..."
    mkdir -p $OUT
    python -m scripts.training_sample --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST --src es --tgt en --save_dir $OUT --batch_size 1 --freeze_strategy fix_nmt_dec_lm_enc --gpus 1 --tokenizer $PRETRAINED_LM_EN --max_input_len 85 --max_output_len 85 --warmup 0 --lr 0.000003 --model_lm_path $PRETRAINED_LM_EN --model_seq2seq_path $PRETRAINED_MT --epochs 10 --seed $SEED > $OUT/log_results.txt 

done
done
