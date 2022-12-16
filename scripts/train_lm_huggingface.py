import pandas as pd
import numpy as np

import torch
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from pathlib import Path


def read_xnli(the_dir, label_dict=None, is_label=True):

    df = pd.read_csv(the_dir, delimiter='\t')
    texts_premise = []
    texts_hypothesis = []
    if is_label:
        labels = []
        if label_dict is None:
            label_dict = {}
    counter = 0
    for i, row in df.iterrows():
        texts_premise.append(row[0])
        texts_hypothesis.append(row[1])
        if is_label:
            if row[2] not in label_dict:
                label_dict[row[2]] = counter
                counter += 1
            labels.append(label_dict[row[2]])

    if is_label:
        return texts_premise, texts_hypothesis, labels, label_dict

    return texts_premise, texts_hypothesis


class XNLIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, tokenizer, is_label=True):
        self.encodings = encodings
        self.labels = labels
        self.is_label = is_label

        # self.my_writer = open('huggingface_input.txt', 'w')
        # self.my_writer_label = open("huggingface_labels.txt", 'w')
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # decoded_sent = [self.tokenizer.decode(t, skip_special_tokens=False) for t in item['input_ids']]
        # self.my_writer.write(" ".join(decoded_sent)+'\n')
        # self.my_writer_label.write(str(int(self.labels[idx])) + '\n')
        if self.is_label:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        if self.is_label:
            return len(self.labels)
        # for key, val in self.encodings.items():
        #     print(key)
        # print(len(self.encodings['input_ids']))
        return len(self.encodings['input_ids'])


def main():

    # Load dataset
    train_texts_prem, train_texts_hyp, train_labels, label_dict = read_xnli('../my_datasets/xnli/train-en.tsv')
    print(label_dict)
    dev_texts_prem, dev_texts_hyp, dev_labels, _ = read_xnli('../my_datasets/xnli/dev-en.tsv', label_dict=label_dict)
    test_texts_prem, test_texts_hyp = read_xnli('../my_datasets/xnli/test-en.tsv', is_label=False)
    dict_label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('../pretrained_lm/bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('../models/huggingdace_lm/checkpoint-122720', num_labels=3)

    # Preprocess_data
    train_encodings = tokenizer(train_texts_prem, train_texts_hyp, truncation=True, padding=True, max_length=128)
    dev_encodings = tokenizer(dev_texts_prem, dev_texts_hyp, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts_prem, test_texts_hyp, truncation=True, padding=True, max_length=128)

    # Create Torch dataset
    train_dataset = XNLIDataset(train_encodings, train_labels, tokenizer)
    dev_dataset = XNLIDataset(dev_encodings, dev_labels, tokenizer)
    test_dataset = XNLIDataset(test_encodings, None, tokenizer, is_label=False)

    training_args = TrainingArguments(
        output_dir='./models',  # output directory
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=10,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        learning_rate=3e-5,
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10000,
    )

    metric = load_metric("accuracy")
    # write_preds = open('huggingface_preds.txt', 'w')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # for elem in predictions:
        #     print(elem)
        #     write_preds.write(str(int(elem))+'\n')
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics
    )

    # trainer.train()
    test_preds = trainer.predict(test_dataset=test_dataset).predictions
    labels_preds = np.argmax(test_preds, 1)
    print('Evaluation results: ', labels_preds)
    print(len(labels_preds))
    write_predictions = open('../models/huggingdace_lm/predictions/xnli/test-en.tsv', 'w')
    for i in range(len(labels_preds)):
        write_predictions.write(dict_label[labels_preds[i]])
        if i < len(labels_preds) - 1:
            write_predictions.write('\n')


if __name__ == '__main__':
    main()
