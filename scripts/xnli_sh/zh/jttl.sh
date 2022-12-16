

for sample in 10 100; do
  TRAIN=my_datasets/sequence_classification/xnli/dev-zh-mini_dev-mini_dev_${sample}.tsv
  DEV=my_datasets/sequence_classification/xnli/dev-zh-mini_test.tsv
  TEST=my_datasets/sequence_classification/xnli/dev-zh-mini_test.tsv


  for SEED in 1 2 3; do
    OUT=models/xnli/jttls/zh/PRETRAINED_mBart_seed_${SEED}_${sample}

    PRETRAINED_LM_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o_seed_$SEED/test/checkpoints*
    PRETRAINED_MT=pretrained_lm/facebook-mbart-large-50-many-to-one-mmt

    echo "Training JTTL..."
    mkdir -p $OUT
    python -m scripts.training_sample --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --src zh --tgt en --batch_size 1 --freeze_strategy fix_nmt_dec_lm_enc --gpus 1 --tokenizer $PRETRAINED_LM_EN --max_input_len 85 --max_output_len 85 --warmup 0 --lr 0.000003 --model_lm_path $PRETRAINED_LM_EN --model_seq2seq_path $PRETRAINED_MT --epochs 10 --seed $SEED > $OUT/log_results.txt

done
done
