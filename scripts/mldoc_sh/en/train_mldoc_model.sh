

TRAIN=my_datasets/sequence_classification/MLdoc_corpus/english/train.10000.txt
DEV=my_datasets/sequence_classification/MLdoc_corpus/english/dev.txt
TEST=my_datasets/sequence_classification/MLdoc_corpus/english/dev.txt

for SEED in 1 2 3; do

  OUT=models/MLdoc/pretrained_lms/en/bert-base-multilingual-cased_seed_$SEED

  PRETRAINED_EN=pretrained_lm/bert-base-multilingual-cased

  echo "Training English MLDOC model $SEED..."
  mkdir -p $OUT
  python -m scripts.train_lm --task MLdoc --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --batch_size 4 --grad_accum 2 --gpus 1 --max_input_len 512 --seed $SEED --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN --epochs 10 > $OUT/log_results.txt

done
