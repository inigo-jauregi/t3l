

TRAIN=my_datasets/sequence_classification/xnli/dev-ar-mini_dev-mini_dev_10.tsv
DEV=my_datasets/sequence_classification/xnli/dev-ar-mini_test.tsv
TEST=my_datasets/sequence_classification/xnli/dev-ar-mini_test.tsv

for SEED in 2 3; do
  OUT=models/xnli/few_shot/ar/mBart_en_ar_50m2o_lr_3e-6_seed_$SEED.samples_10

  PRETRAINED_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o_seed_$SEED/test/checkpoints*
 
  echo "Training few shot..."
  mkdir -p $OUT
  #python -m scripts.train_lm --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --batch_size 8 --grad_accum 2 --lr 0.000003 --gpus 1 --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN --epochs 10 --seed 1234 > $OUT/log_results.txt

  python -m scripts.train_lm --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --batch_size 1 --lr 0.000003 --gpus 1 --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN --epochs 10 --seed $SEED > $OUT/log_results.txt

done

TRAIN=my_datasets/sequence_classification/xnli/dev-ar-mini_dev-mini_dev_100.tsv
DEV=my_datasets/sequence_classification/xnli/dev-ar-mini_test.tsv
TEST=my_datasets/sequence_classification/xnli/dev-ar-mini_test.tsv

for SEED in 2 3; do
  OUT=models/xnli/few_shot/ar/mBart_en_ar_50m2o_lr_3e-6_seed_$SEED.samples_100

  PRETRAINED_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o_seed_$SEED/test/checkpoints*
 
  echo "Training few shot..."
  mkdir -p $OUT
  #python -m scripts.train_lm --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --batch_size 8 --grad_accum 2 --lr 0.000003 --gpus 1 --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN --epochs 10 --seed 1234 > $OUT/log_results.txt

  python -m scripts.train_lm --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --batch_size 8 --grad_accum 2 --lr 0.000003 --gpus 1 --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN --epochs 10 --seed $SEED > $OUT/log_results.txt

done



