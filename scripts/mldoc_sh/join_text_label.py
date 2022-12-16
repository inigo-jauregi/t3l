
reading_file = open('../../my_datasets/sequence_classification/MLdoc_corpus/english/train.10000.it', 'r')
prem_list = []
for line in reading_file:
    line = line.replace('\n', '').strip()
    # Lists
    prem_list.append(line)

reading_file = open('../../my_datasets/sequence_classification/MLdoc_corpus/english/train.10000_label.txt', 'r')
label_list = []
for line in reading_file:
    line = line.replace('\n', '').strip()
    label_list.append(line)


print(len(prem_list))
print(len(label_list))

join_write = open('../../my_datasets/sequence_classification/MLdoc_corpus/english/train.10000-en2it.txt', 'w')
for i in range(len(prem_list)):
    join_write.write(label_list[i] + '\t' + prem_list[i] + '\n')

join_write.close()
