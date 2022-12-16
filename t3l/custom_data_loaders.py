import json

import pandas as pd

import torch
from torch.utils.data import Dataset

# CONSTANT label2id mappings for all datasets
XNLI_LABEL2ID = {'neutral': 1, 'entailment': 0, 'contradiction': 2}
XNLI_ID2LABEL = {1: "neutral", 0: "entailment", 2: 'contradiction'}
MLDOC_LABEL2ID = {'CCAT': 0, 'ECAT': 1, 'GCAT': 2, 'MCAT': 3}
MARC_LABEL2ID = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
MULTIEURLEX_LEVEL_1_LABEL2ID = {'industry': 0, 'environment': 1, 'production, technology and research': 2,
                                'trade': 3, 'social questions': 4, 'finance': 5, 'agri-foodstuffs': 6,
                                'economics': 7, 'agriculture, forestry and fisheries': 8,
                                'education and communications': 9, 'geography': 10, 'business and competition': 11,
                                'international relations': 12, 'politics': 13, 'transport': 14, 'EUROPEAN UNION': 15,
                                'energy': 16, 'employment and working conditions': 17, 'law': 18,
                                'international organisations': 19, 'science': 20}

class XNLIDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_input_len, label_dict):

        # df = pd.read_csv(hf_dataset, delimiter='\t', header=None)
        examples = []
        # for i, row in df.iterrows():
        #     examples.append({'premise': row[0], 'hypothesis': row[1], 'label': row[2]})
        read_file = open(hf_dataset,'r')
        n_error = 0
        for line in read_file:
            row = line.strip().split('\t')
            if len(row) != 3:
                n_error += 1
                continue
            examples.append({'premise': row[0], 'hypothesis': row[1], 'label': row[2]})

        print(label_dict)
        print(f'Number of erroneus examples -> {n_error}')
        self.hf_dataset = examples
        self.label_dict = label_dict
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len

        # self.writer_input = open('lightning_input.txt', 'w')

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        sequence_1_ids = self.tokenizer.encode(entry['premise'], truncation=True, max_length=self.max_input_len)
        sequence_2_ids = self.tokenizer.encode(entry['hypothesis'], truncation=True, max_length=self.max_input_len)
        #
        sequence = {}
        sequence['input_ids'] = [self.tokenizer.cls_token_id] + sequence_1_ids[:-2] + [self.tokenizer.sep_token_id] + \
                                 sequence_2_ids[:-2] + [self.tokenizer.sep_token_id]
        # sequence = self.tokenizer(entry['premise'], entry['hypothesis'], truncation=True, padding=True,
        #                           max_length=self.max_input_len)

        sequence['attention_mask'] = [1] * len(sequence['input_ids'])
        sequence['token_type_ids'] = [0] * (2 + len(sequence_1_ids[:-2])) + [1] * (1 + len(sequence_2_ids[:-2]))

        # print(sequence['input_ids'])
        # translations = [self.tokenizer.decode(t, skip_special_tokens=False) for t in sequence['input_ids']]
        # print(translations)
        # self.writer_input.write(" ".join(translations) + '\n')
        # Check encoded sentence
        output_label = self.label_dict[entry['label']]  # Do I need to one-hot encode the output?

        # if self.tokenizer.bos_token_id is None:  # pegasus
        #     output_ids = [self.tokenizer.pad_token_id] +
        return torch.tensor(sequence['input_ids']), torch.tensor(sequence['attention_mask']), \
               torch.tensor(sequence['token_type_ids']), torch.tensor(output_label)

    @staticmethod
    def collate_fn(batch):
        # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
        # if batch[0][0][-1].item() == 2:
        #     pad_token_id = 1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        # elif batch[0][0][-1].item() == 1:
        #     pad_token_id = 0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        # else:
        #     assert False

        sequence_ids, attention_mask, token_type_ids, labels = list(zip(*batch))
        sequence_ids = torch.nn.utils.rnn.pad_sequence(sequence_ids, batch_first=True, padding_value=1)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        # Maybe one-hot encode the label
        labels = torch.tensor(labels)
        return sequence_ids, attention_mask, token_type_ids, labels


# Variation to be able to translate the premise and hypothesis separatly
# To be used in the translate and test, and jttl cases.
class XNLIDataset_tt(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_input_len, max_output_len, label_dict):

        df = pd.read_csv(hf_dataset, delimiter='\t')
        examples = []
        for i, row in df.iterrows():
            examples.append({'premise': row[0], 'hypothesis': row[1], 'label': row[2]})

        self.hf_dataset = examples
        self.label_dict = label_dict
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        sequence_1_ids = self.tokenizer.encode(entry['premise'].strip())
        sequence_2_ids = self.tokenizer.encode(entry['hypothesis'].strip())

        attention_1_mask = [1] * len(sequence_1_ids)
        attention_2_mask = [1] * len(sequence_2_ids)

        # translations = [self.tokenizer.decode(t, skip_special_tokens=False) for t in sequence_1_ids]
        # print(translations)
        # print(sequence_1_ids)
        # Check encoded sentence
        output_label = self.label_dict[entry['label']]  # Do I need to one-hot encode the output?

        # if self.tokenizer.bos_token_id is None:  # pegasus
        #     output_ids = [self.tokenizer.pad_token_id] +
        return torch.tensor(sequence_1_ids), torch.tensor(sequence_2_ids), torch.tensor(attention_1_mask), \
               torch.tensor(attention_2_mask), torch.tensor(output_label)

    @staticmethod
    def collate_fn(batch):
        # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
        # if batch[0][0][-1].item() == 2:
        #     pad_token_id = 1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        # elif batch[0][0][-1].item() == 1:
        #     pad_token_id = 0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        # else:
        #     assert False

        sequence_1_ids, sequence_2_ids, attention_1_mask, attention_2_mask, labels = list(zip(*batch))
        sequence_ids = sequence_1_ids + sequence_2_ids
        attention_mask = attention_1_mask + attention_2_mask
        sequence_ids = torch.nn.utils.rnn.pad_sequence(sequence_ids, batch_first=True, padding_value=1)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # Maybe one-hot encode the label
        labels = torch.tensor(labels)
        return sequence_ids, attention_mask, labels


class MLDocCorpus(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_input_len, label_dict):

        examples = []
        with open(hf_dataset, encoding='utf-8') as text_reader:
            for line in text_reader:
                line_splited = line.split('\t')
                examples.append({'text': line_splited[1],
                                 'label': line_splited[0]})

        print(label_dict)
        self.hf_dataset = examples
        self.label_dict = label_dict
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len

        # self.writer_input = open('lightning_input.txt', 'w')

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        sequence_ids = self.tokenizer.encode(entry['text'], truncation=True, max_length=self.max_input_len)

        sequence = {}
        sequence['input_ids'] = sequence_ids
        sequence['attention_mask'] = [1] * len(sequence_ids)
        sequence['token_type_ids'] = [1] * len(sequence_ids)

        # print(sequence['input_ids'])
        # translations = [self.tokenizer.decode(t, skip_special_tokens=False) for t in sequence['input_ids']]
        # print(translations)
        # self.writer_input.write(" ".join(translations) + '\n')
        # Check encoded sentence
        output_label = self.label_dict[entry['label']]  # Do I need to one-hot encode the output?

        # if self.tokenizer.bos_token_id is None:  # pegasus
        #     output_ids = [self.tokenizer.pad_token_id] +
        return torch.tensor(sequence['input_ids']), torch.tensor(sequence['attention_mask']), \
               torch.tensor(sequence['token_type_ids']), torch.tensor(output_label)

    @staticmethod
    def collate_fn(batch):
        # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
        # if batch[0][0][-1].item() == 2:
        #     pad_token_id = 1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        # elif batch[0][0][-1].item() == 1:
        #     pad_token_id = 0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        # else:
        #     assert False

        sequence_ids, attention_mask, token_type_ids, labels = list(zip(*batch))
        sequence_ids = torch.nn.utils.rnn.pad_sequence(sequence_ids, batch_first=True, padding_value=1)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        # Maybe one-hot encode the label
        labels = torch.tensor(labels)
        return sequence_ids, attention_mask, token_type_ids, labels


class MLDocCorpus_tt(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_input_len, label_dict):

        examples = []
        with open(hf_dataset, encoding='utf-8') as text_reader:
            for line in text_reader:
                line_splited = line.split('\t')
                examples.append({'text': line_splited[1],
                                 'label': line_splited[0]})

        print(label_dict)
        self.hf_dataset = examples
        self.label_dict = label_dict
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len

        # self.writer_input = open('lightning_input.txt', 'w')

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        sequence_ids = self.tokenizer.encode(entry['text'].strip(), truncation=True, max_length=self.max_input_len)

        sequence = {}
        sequence['input_ids'] = sequence_ids
        sequence['attention_mask'] = [1] * len(sequence_ids)

        # print(sequence['input_ids'])
        # translations = [self.tokenizer.decode(t, skip_special_tokens=False) for t in sequence['input_ids']]
        # print(translations)
        # self.writer_input.write(" ".join(translations) + '\n')
        # Check encoded sentence
        output_label = self.label_dict[entry['label']]  # Do I need to one-hot encode the output?

        # if self.tokenizer.bos_token_id is None:  # pegasus
        #     output_ids = [self.tokenizer.pad_token_id] +
        return torch.tensor(sequence['input_ids']), torch.tensor(sequence['attention_mask']), torch.tensor(output_label)

    @staticmethod
    def collate_fn(batch):
        # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
        # if batch[0][0][-1].item() == 2:
        #     pad_token_id = 1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        # elif batch[0][0][-1].item() == 1:
        #     pad_token_id = 0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        # else:
        #     assert False

        sequence_ids, attention_mask, labels = list(zip(*batch))
        sequence_ids = torch.nn.utils.rnn.pad_sequence(sequence_ids, batch_first=True, padding_value=1)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        # Maybe one-hot encode the label
        labels = torch.tensor(labels)
        return sequence_ids, attention_mask, labels


class MultiEurlexCorpus(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_input_len, label_dict):

        examples = []
        with open(hf_dataset, encoding='utf-8') as json_reader:
            for line in json_reader:
                data_object = json.loads(line)
                examples.append({'text': data_object['text'],
                                 'labels': [MULTIEURLEX_LEVEL_1_LABEL2ID[x['eurovoc_desc_en']]
                                            for x in data_object['labels']]})

        print(label_dict)
        self.hf_dataset = examples
        self.label_dict = label_dict
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len

        # self.writer_input = open('lightning_input.txt', 'w')

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        sequence_ids = self.tokenizer.encode(entry['text'], truncation=True, max_length=self.max_input_len)

        sequence = {}
        sequence['input_ids'] = sequence_ids
        sequence['attention_mask'] = [1] * len(sequence_ids)
        sequence['token_type_ids'] = [1] * len(sequence_ids)

        # print(sequence['input_ids'])
        # translations = [self.tokenizer.decode(t, skip_special_tokens=False) for t in sequence['input_ids']]
        # print(translations)
        # self.writer_input.write(" ".join(translations) + '\n')
        # Check encoded sentence
        output_label = torch.zeros((1, len(self.label_dict)))
        output_label[0, entry['labels']] = 1.0

        # if self.tokenizer.bos_token_id is None:  # pegasus
        #     output_ids = [self.tokenizer.pad_token_id] +
        return torch.tensor(sequence['input_ids']), torch.tensor(sequence['attention_mask']), \
               torch.tensor(sequence['token_type_ids']), output_label

    @staticmethod
    def collate_fn(batch):
        # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
        # if batch[0][0][-1].item() == 2:
        #     pad_token_id = 1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        # elif batch[0][0][-1].item() == 1:
        #     pad_token_id = 0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        # else:
        #     assert False

        sequence_ids, attention_mask, token_type_ids, labels = list(zip(*batch))
        sequence_ids = torch.nn.utils.rnn.pad_sequence(sequence_ids, batch_first=True, padding_value=1)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        # Maybe one-hot encode the label
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0).squeeze(1)
        return sequence_ids, attention_mask, token_type_ids, labels


class MultiEurlexCorpus_tt(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_input_len, label_dict):

        examples = []
        with open(hf_dataset, encoding='utf-8') as json_reader:
            for line in json_reader:
                data_object = json.loads(line)
                examples.append({'text': data_object['text'],
                                 'labels': [MULTIEURLEX_LEVEL_1_LABEL2ID[x['eurovoc_desc_en']]
                                            for x in data_object['labels']]})

        print(label_dict)
        self.hf_dataset = examples
        self.label_dict = label_dict
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len

        # self.writer_input = open('lightning_input.txt', 'w')

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        sequence_ids = self.tokenizer.encode(entry['text'], truncation=True, max_length=self.max_input_len)

        sequence = {}
        sequence['input_ids'] = sequence_ids
        sequence['attention_mask'] = [1] * len(sequence_ids)

        output_label = torch.zeros((1, len(self.label_dict)))
        output_label[0, entry['labels']] = 1.0

        # if self.tokenizer.bos_token_id is None:  # pegasus
        #     output_ids = [self.tokenizer.pad_token_id] +
        return torch.tensor(sequence['input_ids']), torch.tensor(sequence['attention_mask']), output_label

    @staticmethod
    def collate_fn(batch):
        # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
        # if batch[0][0][-1].item() == 2:
        #     pad_token_id = 1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        # elif batch[0][0][-1].item() == 1:
        #     pad_token_id = 0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        # else:
        #     assert False

        sequence_ids, attention_mask,  labels = list(zip(*batch))
        sequence_ids = torch.nn.utils.rnn.pad_sequence(sequence_ids, batch_first=True, padding_value=1)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        # Maybe one-hot encode the label
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0).squeeze(1)
        return sequence_ids, attention_mask, labels