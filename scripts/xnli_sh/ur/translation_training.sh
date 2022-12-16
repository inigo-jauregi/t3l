

TRAIN=my_datasets/translation/IWSLT_2014_TEDtalks/ur-en/train
DEV=my_datasets/translation/IWSLT_2014_TEDtalks/ur-en/test
TEST=my_datasets/translation/IWSLT_2014_TEDtalks/ur-en/test

SRC=ur
TGT=en

OUT=models/iwslt/pretrained_mts/ur-en/ur_en_mbart_pr_model_50_m2o_max_seq_len_85

PRETRAINED_LM=pretrained_lm/facebook-mbart-large-50-many-to-one-mmt

echo "Training translator..."
mkdir -p $OUT
python -m scripts.train_translation --train_data $TRAIN --validation_data $DEV --test_data $TEST --src $SRC --tgt $TGT --save_dir $OUT --tokenizer $PRETRAINED_LM --model_lm_path $PRETRAINED_LM --batch_size 8 --grad_accum 2 --gpus 1 --epochs 10
