

TRAIN=my_datasets/sequence_classification/xnli/dev-th-mini_dev.tsv
DEV=my_datasets/sequence_classification/xnli/dev-th-mini_test.tsv
TEST=my_datasets/sequence_classification/xnli/dev-th-mini_test.tsv

OUT=models/xnli/few_shot/th/mBart_en_th_50m2o_lr_3e-6

PRETRAINED_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o/test/checkpoints_0.8333acc

echo "Training few shot..."
mkdir -p $OUT
python -m scripts.train_lm --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --batch_size 8 --gpus 1 --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN --from_pretrained $OUT --test
