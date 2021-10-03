import pickle as pickle
import os
import pandas as pd
import torch
from ast import literal_eval

from tqdm import tqdm


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


def add_entity_token(row):
    sent = row['sentence']
    se = literal_eval(row['subject_entity'])
    oe = literal_eval(row['object_entity'])

    new_sent = ''
    if se['start_idx'] < oe['start_idx']:
        new_sent += sent[:se['start_idx']]
        new_sent += f'<e1> [{se["type"]}] '
        new_sent += sent[se['start_idx']:se['end_idx'] + 1] + ' </e1> '
        new_sent += sent[se['end_idx'] + 1:oe['start_idx']]
        new_sent += f'<e2> [{oe["type"]}] '
        new_sent += sent[oe['start_idx']:oe['end_idx'] + 1] + ' </e2> '
        new_sent += sent[oe['end_idx'] + 1:]
    else:
        new_sent += sent[:oe['start_idx']]
        new_sent += f'<e2> [{oe["type"]}] '
        new_sent += sent[oe['start_idx']:oe['end_idx'] + 1] + ' </e2> '
        new_sent += sent[oe['end_idx'] + 1:se['start_idx']]
        new_sent += f'<e1> [{se["type"]}] '
        new_sent += sent[se['start_idx']:se['end_idx'] + 1] + ' </e1> '
        new_sent += sent[se['end_idx'] + 1:]

    return new_sent


def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    sentence_list = []
    for _, row in tqdm(dataset.iterrows()):
        sentence_list.append(add_entity_token(row))

    out_dataset = pd.DataFrame({'sentence': sentence_list,
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
    return tokenized_test, test_label


def tokenized_dataset(dataset, tokenizer, args):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""

    e_p_list = []
    for sent in dataset.sentence:
        tokenized_sent = tokenizer.tokenize(sent)

        e11_p = tokenized_sent.index('<e1>')  # the start position of entity1
        e12_p = tokenized_sent.index('</e1>')  # the end position of entity1
        e21_p = tokenized_sent.index('<e2>')  # the start position of entity2
        e22_p = tokenized_sent.index('</e2>')  # the end position of entity2

        # Replace the token
        tokenized_sent[e11_p] = "$"
        tokenized_sent[e12_p] = "$"
        tokenized_sent[e21_p] = "#"
        tokenized_sent[e22_p] = "#"

        # Add 1 because of the [CLS] token
        e11_p += 1
        e12_p += 1
        e21_p += 1
        e22_p += 1

        e_p_list.append([e11_p, e12_p, e21_p, e22_p])

    tokenized_sentences = tokenizer(
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False if 'roberta' in args.model_name else True,
    )

    e1_mask = [[0] * tokenized_sentences['attention_mask'].shape[1]
               for _ in range(tokenized_sentences['attention_mask'].shape[0])]
    e2_mask = [[0] * tokenized_sentences['attention_mask'].shape[1]
               for _ in range(tokenized_sentences['attention_mask'].shape[0])]

    for i, e_p in enumerate(tqdm(e_p_list)):
        for j in range(e_p[0], e_p[1] + 1):
            e1_mask[i][j] = 1
        for j in range(e_p[2], e_p[3] + 1):
            e2_mask[i][j] = 1

    tokenized_sentences['e1_mask'] = torch.tensor(e1_mask, dtype=torch.long)
    tokenized_sentences['e2_mask'] = torch.tensor(e2_mask, dtype=torch.long)

    return tokenized_sentences
