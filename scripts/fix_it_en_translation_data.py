import re


def main():

    src_file_path = '../my_datasets/translation/IWSLT_2014_TEDtalks/it-en/test_joined.it'
    tgt_file_path = '../my_datasets/translation/IWSLT_2014_TEDtalks/it-en/test_joined.en'

    read_file = open(src_file_path, 'r', encoding='utf-8')
    list_src = []
    for line in read_file:
        clean_line = re.sub('<seg id=".*?">','',line)
        clean_line = re.sub('</seg>','', clean_line).strip()
        list_src.append(clean_line)
    read_file.close()

    read_file = open(tgt_file_path, 'r', encoding='utf-8')
    list_tgt = []
    for line in read_file:
        clean_line = re.sub('<seg id=".*?">', '', line)
        clean_line = re.sub('</seg>', '', clean_line).strip()
        list_tgt.append(clean_line)
    read_file.close()

    print(len(list_src))
    print(len(list_tgt))


    writer_file_src = open('../my_datasets/translation/IWSLT_2014_TEDtalks/it-en/test_joined_good.it',
                           'w', encoding='utf-8')
    for sen in list_src:
        writer_file_src.write(sen+'\n')
    writer_file_src.close()
    writer_file_tgt = open('../my_datasets/translation/IWSLT_2014_TEDtalks/it-en/test_joined_good.en',
                           'w', encoding='utf-8')
    for sen in list_tgt:
        writer_file_tgt.write(sen + '\n')





if __name__ == '__main__':
    main()