import re


def main():

    src_file_path = '../my_datasets/translation/IWSLT_2014_TEDtalks/nl-en/nl-en/train.tags.nl-en.nl'
    tgt_file_path = '../my_datasets/translation/IWSLT_2014_TEDtalks/nl-en/nl-en/train.tags.nl-en.en'

    read_file = open(src_file_path, 'r', encoding='utf-8')
    list_src = []
    for line in read_file:
        clean_line = line.strip()
        # if len(clean_line) > 0:
            # if clean_line.startswith('<'):
            #     continue
        list_src.append(clean_line)
    read_file.close()

    read_file = open(tgt_file_path, 'r', encoding='utf-8')
    list_tgt = []
    for line in read_file:
        clean_line = line.strip()
        # if len(clean_line) > 0:
            # if clean_line.startswith('<'):
            #     continue
        list_tgt.append(clean_line)
    read_file.close()

    print(len(list_src))
    print(len(list_tgt))


    writer_file_src = open('../my_datasets/translation/IWSLT_2014_TEDtalks/nl-en/nl-en/train_clean.nl',
                           'w', encoding='utf-8')
    for sen in list_src:
        writer_file_src.write(sen+'\n')
    writer_file_src.close()
    writer_file_tgt = open('../my_datasets/translation/IWSLT_2014_TEDtalks/nl-en/nl-en/train_clean.en',
                           'w', encoding='utf-8')
    for sen in list_tgt:
        writer_file_tgt.write(sen + '\n')





if __name__ == '__main__':
    main()