

TRAIN=my_datasets/sequence_classification/xnli/dev-th-mini_dev.tsv
DEV=my_datasets/sequence_classification/xnli/dev-th-mini_test.tsv
TEST=my_datasets/sequence_classification/xnli/dev-th-mini_test.tsv

for SEED in 2 3; do

  OUT=models/xnli/jttls/th/mBart_th_en_50m2o_batch1_lr3-6_seed_$SEED

  PRETRAINED_LM_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o_seed_$SEED/test/checkpoints*
  PRETRAINED_MT=models/iwslt/pretrained_mts/th-en/th_en_mbart_pr_model_50_m2o_max_seq_len_85/test/checkpoints_18.37BLEU

  echo "Training JTTL..."
  mkdir -p $OUT
  python -m scripts.training_sample --train_data $TRAIN --validation_data $DEV --test_data $TEST --src th --tgt en --save_dir $OUT --batch_size 1 --freeze_strategy fix_nmt_dec_lm_enc --gpus 1 --tokenizer $PRETRAINED_LM_EN --max_input_len 85 --max_output_len 85 --warmup 0 --lr 0.000003 --model_lm_path $PRETRAINED_LM_EN --model_seq2seq_path $PRETRAINED_MT --epochs 10 --seed $SEED > $OUT/log_results.txt

done

#echo "Testing JTTL..."
#OUT=models/xnli/jttls/th/placeholder
#python -m scripts.training_sample --train_data $TRAIN --validation_data $DEV --test_data $TEST --src th --tgt en --save_dir $OUT --batch_size 1 --freeze_strategy fix_nmt_dec_lm_enc --gpus 1 --tokenizer $PRETRAINED_LM_EN --max_input_len 85 --max_output_len 85 --warmup 0 --lr 0.000003 --model_lm_path $PRETRAINED_LM_EN --model_seq2seq_path $PRETRAINED_MT --test --from_pretrained models/xnli/jttls/th/mBart_th_en_50m2o_batch1_lr3-6/test/checkpoints/check-epoch=07-avg_val_accuracy=0.70.ckpt
