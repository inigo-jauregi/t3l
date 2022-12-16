

reading_file = open('../../my_datasets/sequence_classification/xnli/dev-en-prem.el', 'r')
prem_list = []
for line in reading_file:
    line = line.replace('\n', '').strip()
    # Lists
    prem_list.append(line)

reading_file = open('../../my_datasets/sequence_classification/xnli/dev-en-hyp.el', 'r')
hyp_list = []
for line in reading_file:
    line = line.replace('\n', '').strip()
    hyp_list.append(line)

reading_file = open('../../my_datasets/sequence_classification/xnli/dev-en-label.tsv', 'r')
label_list = []
for line in reading_file:
    line = line.replace('\n', '').strip()
    label_list.append(line)


print(len(prem_list))
print(len(hyp_list))
print(len(label_list))

join_write = open('../../my_datasets/sequence_classification/xnli/dev-en2el_pretrained.tsv', 'w')
for i in range(len(prem_list)):
    join_write.write(prem_list[i] + '\t' + hyp_list[i] + '\t' + label_list[i] + '\n')

join_write.close()
