import pickle as pickle
import os
import pandas as pd
import torch
from ast import literal_eval


class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
        i = literal_eval(i)['word']
        j = literal_eval(j)['word']

        subject_entity.append(i)
        object_entity.append(j)

    out_dataset = pd.DataFrame({'id': dataset['id'],
                                'sentence': dataset['sentence'],
                                'subject_entity': subject_entity,
                                'object_entity': object_entity,
                                'label': dataset['label'], })

    return out_dataset


def preprocessing_test_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
        i = literal_eval(i)['word']
        j = literal_eval(j)['word']

        subject_entity.append(i)
        object_entity.append(j)

    out_dataset = pd.DataFrame({'id': dataset['id'],
                                'sentence': dataset['sentence'],
                                'subject_entity': subject_entity,
                                'object_entity': object_entity,
                                'label': dataset['label'], })
    return out_dataset


def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


def load_test_dataset(test_dataset, tokenizer, args):
    """
      test dataset을 불러온 후,
      tokenizing 합니다.
    """
    test_label = list(map(int, test_dataset['label'].values))
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer, args)
    return test_dataset['id'], tokenized_test, test_label


def tokenized_dataset(dataset, tokenizer, args):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = f'{e01} 와 {e02} 의 관계는?'
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False if 'roberta' in args.model_name else True,
    )
    return tokenized_sentences
