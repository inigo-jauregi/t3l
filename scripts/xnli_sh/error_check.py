
read_file = open('../../my_datasets/sequence_classification/xnli/dev-en2sw.tsv')

for line in read_file:

    line = line.strip()
    line_split = line.split('\t')
    if len(line_split) != 3:
        print(line)
        print(line_split)
        print(len(line_split))