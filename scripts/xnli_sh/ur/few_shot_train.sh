
for SAMPLE in 10; do

  TRAIN=my_datasets/sequence_classification/xnli/dev-ur-mini_dev-mini_dev_$SAMPLE.tsv
  DEV=my_datasets/sequence_classification/xnli/dev-ur-mini_test.tsv
  TEST=my_datasets/sequence_classification/xnli/test-ur.tsv


  for SEED in 1 2 3; do

    OUT=models/xnli/few_shot/ur/mBart_en_ur_50m2o_lr_3e-6_seed_$SEED.SAMPLES_$SAMPLE

    PRETRAINED_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o_seed_$SEED/test/checkpoints*

    echo "Training few shot..."
    mkdir -p $OUT
    python -m scripts.train_lm --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --batch_size 1 --lr 0.000003 --gpus 1 --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN --epochs 10 --seed $SEED > $OUT/log_results.txt

done
done

for SAMPLE in 100; do

  TRAIN=my_datasets/sequence_classification/xnli/dev-ur-mini_dev-mini_dev_$SAMPLE.tsv
  DEV=my_datasets/sequence_classification/xnli/dev-ur-mini_test.tsv
  TEST=my_datasets/sequence_classification/xnli/test-ur.tsv


  for SEED in 1 2 3; do

    OUT=models/xnli/few_shot/ur/mBart_en_ur_50m2o_lr_3e-6_seed_$SEED.SAMPLES_$SAMPLE

    PRETRAINED_EN=models/xnli/pretrained_lms/en/mBART_large_nmt_50_m2o_seed_$SEED/test/checkpoints*

    echo "Training few shot..."
    mkdir -p $OUT
    python -m scripts.train_lm --task XNLI --train_data $TRAIN --validation_data $DEV --test_data $TEST --save_dir $OUT --batch_size 8 --grad_accum 2 --lr 0.000003 --gpus 1 --tokenizer $PRETRAINED_EN --model_lm_path $PRETRAINED_EN --epochs 10 --seed $SEED > $OUT/log_results.txt

done
done
