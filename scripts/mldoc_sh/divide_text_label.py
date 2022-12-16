from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('../../pretrained_lm/facebook-mbart-large-50-one-to-many-mmt', use_fast=False)

reading_file = open('../../my_datasets/sequence_classification/MLdoc_corpus/english/train.10000.txt', 'r')
prem_list = []
label_list = []
for line in reading_file:
    line = line.replace('\n', '')
    line_split = line.split('\t')
    # Lists
    prem_list.append(line_split[1])
    label_list.append(line_split[0])


print(len(prem_list))
print(len(label_list))

prem_write = open('../../my_datasets/sequence_classification/MLdoc_corpus/english/train.10000.en', 'w')
label_write = open('../../my_datasets/sequence_classification/MLdoc_corpus/english/train.10000_label.txt', 'w')

counter = 0
for i in range(len(prem_list)):
    if len(tokenizer.encode(prem_list[i].strip())) <= 512:
        prem_write.write(prem_list[i] + '\n')
        label_write.write(label_list[i] + '\n')
        counter += 1

print(counter)
prem_write.close()
label_write.close()
