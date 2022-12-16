import pandas as pd

# SRC
src_reader = open('../my_datasets/sequence_classification/xnli/test-es.tsv')
src_list = []
src_label = []
for line in src_reader:
    line_clean = line.replace('\n','')
    line_elems = line_clean.split('\t')
    src_list.append(" HYPOTHESIS: ".join(line_elems[0:2]))
    src_label.append(line_elems[2])

# PRE-TRAINED
examples_pre_trained_reader = open("../models/xnli/intermediate_translations/es/pre_trained/"
                                   "mBart_es_en_50m2o_batch1_lr3-6_seed_1/int_translations.txt")
pre_trained_list = []
for line in examples_pre_trained_reader :
    line_clean = line.replace('\n','')
    pre_trained_list.append(line_clean)
labels_pre_trained_reader = open("../models/xnli/intermediate_translations/es/pre_trained/"
                                   "mBart_es_en_50m2o_batch1_lr3-6_seed_1/pred_labels.txt")
labels_pre_trained_list = []
for line in labels_pre_trained_reader:
    line_clean = line.replace('\n', '')
    labels_pre_trained_list.append(line_clean)

# ZERO-SHOT
examples_zero_shot_reader = open("../models/xnli/intermediate_translations/es/zero_shot/"
                                 "mBart_es_en_50m2o_batch1_lr3-6_seed_1/int_translations.txt")
zero_shot_list = []
for line in examples_zero_shot_reader :
    line_clean = line.replace('\n','')
    zero_shot_list.append(line_clean)
labels_zero_shot_reader = open("../models/xnli/intermediate_translations/es/zero_shot/"
                               "mBart_es_en_50m2o_batch1_lr3-6_seed_1/pred_labels.txt")
labels_zero_shot_list = []
for line in labels_zero_shot_reader:
    line_clean = line.replace('\n', '')
    labels_zero_shot_list.append(line_clean)

# T3L
examples_jttl_reader = open("../models/xnli/intermediate_translations/es/jttl/"
                            "mBart_es_en_50m2o_batch1_lr3-6_seed_1.samples_100/int_translations.txt")
jttl_list = []
for line in examples_jttl_reader :
    line_clean = line.replace('\n','')
    jttl_list.append(line_clean)
labels_jttl_reader = open("../models/xnli/intermediate_translations/es/jttl/"
                                   "mBart_es_en_50m2o_batch1_lr3-6_seed_1.samples_100/pred_labels.txt")
labels_jttl_list = []
for line in labels_jttl_reader:
    line_clean = line.replace('\n', '')
    labels_jttl_list.append(line_clean)

print(len(src_list[1:]))
print(len(src_label[1:]))
print(len(pre_trained_list))
print(len(labels_pre_trained_list))
print(len(zero_shot_list))
print(len(labels_zero_shot_list))
print(len(jttl_list))
print(len(labels_jttl_list))

df = pd.DataFrame()
df['src_example'] = src_list[1:]
df['pre_trained_example'] = pre_trained_list
df['zero_shot_example'] = zero_shot_list
df['jttl_example'] = jttl_list
df['src_label'] = src_label[1:]
df['pre_trained_label'] = labels_pre_trained_list
df['zero_shot_label'] = labels_zero_shot_list
df['jttl_label'] = labels_jttl_list

df.to_csv('../models/xnli/intermediate_translations/es/comparison.csv', index=False, encoding='utf-8')
