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
        new_sent += '<e1> <e3> PER </e3> ' if se['type'] == 'PER' else \
            '<e1> <e3> ORG </e3> ' if se['type'] == 'ORG' else \
                '<e1> <e3> DAT </e3> ' if se['type'] == 'DAT' else \
                    '<e1> <e3> LOC </e3> ' if se['type'] == 'LOC' else \
                        '<e1> <e3> NOH </e3> ' if se['type'] == 'NOH' else '<e1> <e3> POH </e3> '
        new_sent += sent[se['start_idx']:se['end_idx'] + 1] + ' </e1> '
        new_sent += sent[se['end_idx'] + 1:oe['start_idx']]
        new_sent += '<e2> <e4> PER </e4> ' if oe['type'] == 'PER' else \
            '<e2> <e4> ORG </e4> ' if oe['type'] == 'ORG' else \
                '<e2> <e4> DAT </e4> ' if oe['type'] == 'DAT' else \
                    '<e2> <e4> LOC </e4> ' if oe['type'] == 'LOC' else \
                        '<e2> <e4> NOH </e4> ' if oe['type'] == 'NOH' else '<e2> <e4> POH </e4> '
        new_sent += sent[oe['start_idx']:oe['end_idx'] + 1] + ' </e2> '
        new_sent += sent[oe['end_idx'] + 1:]
    else:
        new_sent += sent[:oe['start_idx']]
        new_sent += '<e2> <e4> PER </e4> ' if oe['type'] == 'PER' else \
            '<e2> <e4> ORG </e4> ' if oe['type'] == 'ORG' else \
                '<e2> <e4> DAT </e4> ' if oe['type'] == 'DAT' else \
                    '<e2> <e4> LOC </e4> ' if oe['type'] == 'LOC' else \
                        '<e2> <e4> NOH </e4> ' if oe['type'] == 'NOH' else '<e2> <e4> POH </e4> '
        new_sent += sent[oe['start_idx']:oe['end_idx'] + 1] + ' </e2> '
        new_sent += sent[oe['end_idx'] + 1:se['start_idx']]
        new_sent += '<e1> <e3> PER </e3> ' if se['type'] == 'PER' else \
            '<e1> <e3> ORG </e3> ' if se['type'] == 'ORG' else \
                '<e1> <e3> DAT </e3> ' if se['type'] == 'DAT' else \
                    '<e1> <e3> LOC </e3> ' if se['type'] == 'LOC' else \
                        '<e1> <e3> NOH </e3> ' if se['type'] == 'NOH' else '<e1> <e3> POH </e3> '
        new_sent += sent[se['start_idx']:se['end_idx'] + 1] + ' </e1> '
        new_sent += sent[se['end_idx'] + 1:]

    return new_sent


def preprocessing_dataset(dataset, args):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    if args.tem:
        sentence_list = []
        subject_entity = []
        object_entity = []
        for _, row in tqdm(dataset.iterrows()):
            sentence_list.append(add_entity_token(row))
            subject_entity.append(eval(row['subject_entity'])['word'])
            object_entity.append(eval(row['object_entity'])['word'])

        out_dataset = pd.DataFrame({'sentence': sentence_list,
                                    'subject_entity': subject_entity,
                                    'object_entity': object_entity,
                                    'label': dataset['label'], })
    else:
        subject_entity = []
        object_entity = []
        for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
            i = i[1:-1].split(',')[0].split(':')[1]
            j = j[1:-1].split(',')[0].split(':')[1]

            subject_entity.append(i)
            object_entity.append(j)
        out_dataset = pd.DataFrame(
            {'id': dataset['id'], 'sentence': dataset['sentence'], 'subject_entity': subject_entity,
             'object_entity': object_entity, 'label': dataset['label'], })

    return out_dataset


def load_data(dataset_dir, args):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset, args)

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
    if args.tem:
        # typed entity marker 적용
        e_p_list = []
        for sent in dataset.sentence:
            tokenized_sent = tokenizer.tokenize(sent)

            e11_p = tokenized_sent.index('<e1>')  # the start position of entity1
            e12_p = tokenized_sent.index('</e1>')  # the end position of entity1
            e21_p = tokenized_sent.index('<e2>')  # the start position of entity2
            e22_p = tokenized_sent.index('</e2>')  # the end position of entity2
            e31_p = tokenized_sent.index('<e3>')  # the start position of entity3
            e32_p = tokenized_sent.index('</e3>')  # the end position of entity3
            e41_p = tokenized_sent.index('<e4>')  # the start position of entity4
            e42_p = tokenized_sent.index('</e4>')  # the end position of entity4

            # Replace the token
            tokenized_sent[e11_p] = "@"
            tokenized_sent[e12_p] = "@"
            tokenized_sent[e21_p] = "#"
            tokenized_sent[e22_p] = "#"
            tokenized_sent[e31_p] = "*"
            tokenized_sent[e32_p] = "*"
            tokenized_sent[e41_p] = "∧"
            tokenized_sent[e42_p] = "∧"

            # Add 1 because of the [CLS] token
            e11_p += 1
            e12_p += 1
            e21_p += 1
            e22_p += 1
            e31_p += 1
            e32_p += 1
            e41_p += 1
            e42_p += 1

            e_p_list.append([e11_p, e12_p, e21_p, e22_p, e31_p, e32_p, e41_p, e42_p])

        tokenized_sentences = tokenizer(
            list(dataset['sentence']),
            # concat_entity,
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
        e3_mask = [[0] * tokenized_sentences['attention_mask'].shape[1]
                   for _ in range(tokenized_sentences['attention_mask'].shape[0])]
        e4_mask = [[0] * tokenized_sentences['attention_mask'].shape[1]
                   for _ in range(tokenized_sentences['attention_mask'].shape[0])]

        for i, e_p in enumerate(tqdm(e_p_list)):
            # '#', '*', '@', '∧' 토큰 output vector 만을 사용하는 방법
            e1_mask[i][e_p[0]] = 1
            e1_mask[i][e_p[1]] = 1
            e2_mask[i][e_p[2]] = 1
            e2_mask[i][e_p[3]] = 1
            e3_mask[i][e_p[4]] = 1
            e3_mask[i][e_p[5]] = 1
            e4_mask[i][e_p[6]] = 1
            e4_mask[i][e_p[7]] = 1

        tokenized_sentences['e1_mask'] = torch.tensor(e1_mask, dtype=torch.long)
        tokenized_sentences['e2_mask'] = torch.tensor(e2_mask, dtype=torch.long)
        # tokenized_sentences['e3_mask'] = torch.tensor(e3_mask, dtype=torch.long)
        # tokenized_sentences['e4_mask'] = torch.tensor(e4_mask, dtype=torch.long)
    else:
        # baseline code
        concat_entity = []
        for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
            temp = ''
            temp = e01 + '[SEP]' + e02
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

