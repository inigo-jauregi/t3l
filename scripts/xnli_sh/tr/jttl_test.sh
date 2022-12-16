

TRAIN=my_datasets/sequence_classification/xnli/dev-tr-mini_dev.tsv
DEV=my_datasets/sequence_classification/xnli/dev-tr-mini_test.tsv
TEST=my_datasets/sequence_classification/xnli/test-tr.tsv

OUT=models/xnli/jttls/tr/placeholder
FROM_PRETRAINED=models/xnli/jttls/tr/mBart_tr_en_50m2o_batch1_lr3-6/test/checkpoints/check-epoch=01-avg_val_accuracy=0.72.ckpt

PRETRAINED_LM_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o/test/checkpoints_0.8333acc
PRETRAINED_MT=models/iwslt/pretrained_mts/tr-en/tr_en_mbart_pr_model_50_m2o_max_seq_len_85/test/checkpoints_44.86BLEU

echo "Testing JTTL..."
python -m scripts.training_sample --train_data $TRAIN --validation_data $DEV --test_data $TEST --src th --tgt en --save_dir $OUT --batch_size 8 --freeze_strategy fix_nmt_dec_lm_enc --gpus 1 --tokenizer $PRETRAINED_LM_EN --max_input_len 85 --max_output_len 85 --warmup 0 --lr 0.000003 --model_lm_path $PRETRAINED_LM_EN --model_seq2seq_path $PRETRAINED_MT --test --from_pretrained $FROM_PRETRAINED
