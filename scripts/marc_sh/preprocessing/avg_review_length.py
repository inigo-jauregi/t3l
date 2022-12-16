import json
from tqdm import tqdm
from transformers import AutoTokenizer

def main():

    dataset = '../../../my_datasets/sequence_classification/MARC_corpus/json/dev/dataset_de_dev.json'
    tokenizer_path = '../../../pretrained_lm/facebook-mbart-large-50-many-to-one-mmt'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    length_title = []
    length_body = []
    length_both = []
    count = 0

    with open(dataset, encoding='utf-8') as json_reader:
        for line in tqdm(json_reader):
            data_object = json.loads(line)

            title = data_object['review_title']
            body = data_object['review_body']

            tok_title = tokenizer.encode(title)
            tok_body = tokenizer.encode(body)

            len_title = len(tok_title)
            len_body = len(tok_body)
            len_both = len_title + len_body

            length_title.append(len_title)
            length_body.append(len_body)
            length_both.append(len_both)

            if len_both > 256:
                count+=1

    print('RESULTS:')
    print('TITLE')
    print('Average: ', sum(length_title)/len(length_title))
    print('MAX: ', max(length_title))
    print('MIN: ', min(length_title))
    print('BODY')
    print('Average: ', sum(length_body) / len(length_body))
    print('MAX: ', max(length_body))
    print('MIN: ', min(length_body))
    print('BOTH')
    print('Average: ', sum(length_both) / len(length_both))
    print('MAX: ', max(length_both))
    print('MIN: ', min(length_both))
    print('')
    print('Num above threshold-> ' + str(count))


if __name__ == '__main__':
    main()