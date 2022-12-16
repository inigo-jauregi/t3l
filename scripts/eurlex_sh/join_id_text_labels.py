import json

reading_file = open('../../my_datasets/sequence_classification/MULTI_EURLEX/level_1/en/validation_id.txt', 'r')
id_list = []
for line in reading_file:
    line = line.replace('\n', '').strip()
    # Lists
    id_list.append(line)

reading_file = open('../../my_datasets/sequence_classification/MULTI_EURLEX/level_1/en/validation.pt', 'r')
text_list = []
for line in reading_file:
    line = line.replace('\n', '').strip()
    text_list.append(line)

reading_file = open('../../my_datasets/sequence_classification/MULTI_EURLEX/level_1/en/validation_label.json', 'r')
label_list = []
for line in reading_file:
    data_obj = json.loads(line)
    label_list.append(data_obj)


print(len(id_list))
print(len(text_list))
print(len(label_list))

join_write = open('../../my_datasets/sequence_classification/MULTI_EURLEX/level_1/en/validation-en2pt_pretrained.json', 'w',
                  encoding='utf-8')
for i in range(len(id_list)):
    out_data = {'celex_id': id_list[i],
                'text': text_list[i],
                'labels': label_list[i]['labels']}
    join_write.write(json.dumps(out_data) + '\n')

join_write.close()
