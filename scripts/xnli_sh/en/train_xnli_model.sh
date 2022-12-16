

TRAIN=my_datasets/sequence_classification/xnli/train-en.tsv
DEV=my_datasets/sequence_classification/xnli/dev-en.tsv
TEST=my_datasets/sequence_classification/xnli/dev-en.tsv

for SEED_NUM in 1 2 3; do

  OUT=models/xnli/pretrained_lms/en/XLM-R_large_seed_$SEED_NUM

  PRETRAINED_EN=pretrained_lm/xlm-roberta-large

  echo "Training English XNLI model..."
  mkdir -p $OUT
  python -m scripts.train_lm --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --batch_size 8 --grad_accum 2 --lr 0.000003 --gpus 1 --seed $SEED_NUM --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN --epochs 10

done

#--task XNLI --train_data ../my_datasets/sequence_classification/xnli/train-en.tsv --validation_data ../my_datasets/sequence_classification/xnli/dev-en.tsv --test_data ../my_datasets/sequence_classification/xnli/dev-en.tsv --save_dir ../models/xnli/pretrained_lms/en/XLM-R_large_seed_1 --batch_size 8 --grad_accum 2 --gpus 1 --seed 1 --tokenizer ../pretrained_lm/xlm-roberta-large --model_lm_path ../pretrained_lm/xlm-roberta-large --epochs 10
