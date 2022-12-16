

TRAIN=my_datasets/sequence_classification/xnli/dev-zh-mini_dev.tsv
DEV=my_datasets/sequence_classification/xnli/dev-zh-mini_test.tsv
TEST=my_datasets/sequence_classification/xnli/dev-zh-mini_test.tsv

OUT=models/xnli/jttls/zh/placeholder
FROM_PRETRAINED=models/xnli/jttls/zh/mBart_mt_19BLEU_en_zh_lr_3e-6_batch1_warmup_0/test/checkpoints/check-epoch=06-avg_val_accuracy=0.72.ckpt

PRETRAINED_LM_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o/test/checkpoints_0.8333acc
PRETRAINED_MT=models/iwslt/pretrained_mts/zh-en/zh_en_mbart_50_m2o_max_seq_len_85/test/checkpoints_19.075996931200997

echo "Testing JTTL..."
python -m scripts.training_sample --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --src zh --tgt en --batch_size 1 --freeze_strategy fix_nmt_dec_lm_enc --gpus 1 --tokenizer $PRETRAINED_LM_EN --max_input_len 85 --max_output_len 85 --model_lm_path $PRETRAINED_LM_EN --model_seq2seq_path $PRETRAINED_MT --test --from_pretrained $FROM_PRETRAINED


