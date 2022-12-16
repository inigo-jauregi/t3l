

TRAIN=my_datasets/sequence_classification/MARC_corpus/json/train/dataset_en_train.json
DEV=my_datasets/sequence_classification/MARC_corpus/json/dev/dataset_en_dev.json
TEST=my_datasets/sequence_classification/MARC_corpus/json/dev/dataset_en_dev.json

for SEED in 1; do

  OUT=models/MARC/pretrained_lms/en/mBART_large_nmt_50_m2o_max_input_len_128_seed_$SEED.ANOTHER

  PRETRAINED_EN=pretrained_lm/facebook-mbart-large-50-many-to-one-mmt

  echo "Training English MLDOC model $SEED..."
  mkdir -p $OUT
  python -m scripts.train_lm --task MARC --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --batch_size 8 --grad_accum 2 --gpus 1 --max_input_len 128 --seed $SEED --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN --epochs 10 --seed $SEED > $OUT/log_results.txt

done
