import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('../../pretrained_lm/facebook-mbart-large-50-one-to-many-mmt', use_fast=False)

reading_file = open('../../my_datasets/sequence_classification/MULTI_EURLEX/level_1/en/train.json', 'r')
id_list = []
text_list = []
label_list = []
for line in reading_file:
    data_object = json.loads(line)
    # Lists
    id_list.append(data_object['celex_id'])
    text_list.append(data_object['text'].replace('\n', ' '))
    label_list.append(data_object['labels'])


print(len(id_list))
print(len(text_list))
print(len(label_list))

id_write = open('../../my_datasets/sequence_classification/MULTI_EURLEX/level_1/en/train_id.txt', 'w')
text_write = open('../../my_datasets/sequence_classification/MULTI_EURLEX/level_1/en/train.en', 'w')
label_write = open('../../my_datasets/sequence_classification/MULTI_EURLEX/level_1/en/train_label.json', 'w')

counter = 0
for i in range(len(id_list)):
    if len(tokenizer.encode(text_list[i].strip())) <= 512:
        id_write.write(id_list[i] + '\n')
        text_write.write(text_list[i] + '\n')
        my_dict = {'labels': label_list[i]}
        label_write.write(json.dumps(my_dict) + '\n')
        counter += 1

print(counter)
id_write.close()
text_write.close()
label_write.close()