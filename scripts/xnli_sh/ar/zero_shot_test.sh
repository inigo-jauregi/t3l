

TRAIN=my_datasets/sequence_classification/xnli/dev-ar-mini_dev.tsv
DEV=my_datasets/sequence_classification/xnli/dev-ar-mini_test.tsv
TEST=my_datasets/sequence_classification/xnli/dev-ar-mini_test.tsv

SEED=1
OUT=models/xnli/zero_shot/ar/mBart_50m2o_seed_$SEED

PRETRAINED_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o_seed_$SEED/test/checkpoints*

echo "Testing zero shot..."
mkdir -p $OUT
python -m scripts.train_lm --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --batch_size 8 --grad_accum 2 --gpus 1 --seed $SEED --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN --test > $OUT/log_results.txt
