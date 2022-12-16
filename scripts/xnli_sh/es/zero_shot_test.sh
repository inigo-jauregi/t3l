

TRAIN=my_datasets/sequence_classification/xnli/dev-es-mini_dev.tsv
DEV=my_datasets/sequence_classification/xnli/dev-es-mini_test.tsv
TEST=my_datasets/sequence_classification/xnli/dev-es-mini_test.tsv

OUT=models/xnli/zero_shot/es/mBart_large_en

PRETRAINED_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o_seed_1/test/checkpoints*

echo "Testing zero shot..."
mkdir -p $OUT
python -m scripts.train_lm --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --batch_size 8 --grad_accum 2 --gpus 1 --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN --test --seed 1234 > $OUT/log_results.txt

for SEED in 2 3; do

  OUT=models/xnli/zero_shot/es/mBart_large_en_seed_$SEED

  PRETRAINED_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o_seed_$SEED/test/checkpoints*

  echo "Testing zero shot..."
  mkdir -p $OUT
  python -m scripts.train_lm --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --batch_size 8 --grad_accum 2 --gpus 1 --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN --test --seed $SEED > $OUT/log_results.txt

done