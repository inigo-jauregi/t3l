from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('../../pretrained_lm/facebook-mbart-large-50-one-to-many-mmt', use_fast=False)

reading_file = open('../../my_datasets/sequence_classification/xnli/train-en.tsv', 'r')
prem_list = []
hyp_list = []
label_list = []
for line in reading_file:
    line = line.replace('\n', '')
    line_split = line.split('\t')
    # Lists
    prem_list.append(line_split[0])
    hyp_list.append(line_split[1])
    label_list.append(line_split[2])


print(len(prem_list))
print(len(hyp_list))
print(len(label_list))

prem_write = open('../../my_datasets/sequence_classification/xnli/train-en-prem.tsv', 'w')
hyp_write = open('../../my_datasets/sequence_classification/xnli/train-en-hyp.tsv', 'w')
label_write = open('../../my_datasets/sequence_classification/xnli/train-en-label.tsv', 'w')

counter = 0
for i in range(len(prem_list)):
    if len(tokenizer.encode(prem_list[i].strip())) <= 170 and len(tokenizer.encode(hyp_list[i].strip())) <= 170:
        prem_write.write(prem_list[i] + '\n')
        hyp_write.write(hyp_list[i] + '\n')
        label_write.write(label_list[i] + '\n')
        counter += 1

print(counter)
prem_write.close()
hyp_write.close()
label_write.close()
