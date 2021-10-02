import argparse
import pickle as pickle
import os
import re
import pandas as pd
import torch
import random
import sklearn
import numpy as np
from glob import glob
from torch import nn
from torch.optim import Optimizer
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer,DataCollatorForLanguageModeling, AutoConfig, AutoModelForMaskedLM, AutoModelForSequenceClassification, Trainer, TrainingArguments,AdamW
    # RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data_XL import *


# 학습한 모델을 재생산하기 위해 seed를 고정
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# 경로

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
                  'org:product', 'per:title', 'org:alternate_names',
                  'per:employee_of', 'org:place_of_headquarters', 'per:product',
                  'org:number_of_employees/members', 'per:children',
                  'per:place_of_residence', 'per:alternate_names',
                  'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
                  'per:spouse', 'org:founded', 'org:political/religious_affiliation',
                  'org:member_of', 'per:parents', 'org:dissolved',
                  'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
                  'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
                  'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0
    # return sklearn.metrics.f1_score(labels, preds, average="micro") * 100.0



def klue_re_auprc(probs, labels, num_labels=30):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(num_labels)[labels]

    score = np.zeros((num_labels,))
    for c in range(num_labels):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    num_labels = len(np.unique(labels))
    if num_labels == 30:
        f1 = klue_re_micro_f1(preds, labels)
        auprc = klue_re_auprc(probs, labels)
        acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.
    elif num_labels == 3:
        f1 = klue_re_micro_f1(preds, labels)
        auprc = klue_re_auprc(probs, labels, num_labels)
        acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.
    elif num_labels == 11:
        f1 = klue_re_micro_f1(preds, labels)
        auprc = klue_re_auprc(probs, labels, num_labels)
        acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.
    elif num_labels == 18:
        f1 = klue_re_micro_f1(preds, labels)
        auprc = klue_re_auprc(probs, labels, num_labels)
        acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.
    else:
        raise Exception('default, nop, org, per 중의 train_type 을 넣어주세요!')

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
    }


def label_to_num(label, args):
    num_label = []
    if args.train_type == 'default':
        with open('dict_label_to_num.pkl', 'rb') as f:
            dict_label_to_num = pickle.load(f)
    elif args.train_type == 'nop':
        dict_label_to_num = {'no_relation': 0,
                             'org': 1,
                             'per': 2}
    elif args.train_type == 'org':
        dict_label_to_num = {'number_of_employees/members': 0,
                             'dissolved': 1,
                             'political/religious_affiliation': 2,
                             'founded_by': 3,
                             'product': 4,
                             'members': 5,
                             'founded': 6,
                             'place_of_headquarters': 7,
                             'alternate_names': 8,
                             'member_of': 9,
                             'top_members/employees': 10}
    elif args.train_type == 'per':
        dict_label_to_num = {'place_of_death': 0,
                             'schools_attended': 1,
                             'religion': 2,
                             'siblings': 3,
                             'product': 4,
                             'place_of_birth': 5,
                             'other_family': 6,
                             'place_of_residence': 7,
                             'children': 8,
                             'date_of_death': 9,
                             'parents': 10,
                             'colleagues': 11,
                             'spouse': 12,
                             'alternate_names': 13,
                             'date_of_birth': 14,
                             'origin': 15,
                             'title': 16,
                             'employee_of': 17}
    else:
        Exception('default, nop, org, per 중의 train_type 을 넣어주세요!')

    for v in label:
        num_label.append(dict_label_to_num[v])


    return num_label


def train(train_df, valid_df, train_label, valid_label, args):
    # load model and tokenizer
    MODEL_NAME = args.model_name
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_df, tokenizer, args)
    tokenized_valid = tokenized_dataset(valid_df, tokenizer, args)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    if args.train_type == 'default':
        model_config.num_labels = 30
    elif args.train_type == 'nop':
        model_config.num_labels = 3
    elif args.train_type == 'org':
        model_config.num_labels = 11
    elif args.train_type == 'per':
        model_config.num_labels = 18

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    print(model.config)
    # model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        save_total_limit=args.save_total_limit,  # number of total save model.
        save_steps=args.save_steps,  # model saving step.
        num_train_epochs=args.epochs,  # total number of training epochs
        learning_rate=args.learning_rate,  # learning_rate
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.valid_batch_size,  # batch size for evaluation
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir=args.logging_dir,  # directory for storing logs
        logging_steps=args.logging_steps,  # log saving step.
        evaluation_strategy=args.evaluation_strategy,  # evaluation strategy to adopt during training
        eval_steps=args.eval_steps,  # evaluation step.
        load_best_model_at_end=args.load_best_model_at_end,
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        # optimizers = torch.optim.AdamW(params=nn.parameters,lr=args.learning_rate, weight_decay=args.weight_decay)
    )

    # train model
    trainer.train()
    model.save_pretrained('./best_model')


def main(args):
    seed_everything(args.seed)

    train_dataset = load_data("../dataset/train/train.csv", args) #load_data_XL 안에 있는 load_data
    # dataset_train, dataset_val = train_test_split(dataset,test_size = 0.2,random_state = args.seed)
    # data_train = BERTDataset(dataset_train, "title", "topic_idx", tokenizer)
    # data_val = BERTDataset(dataset_val, "title", "topic_idx", tokenizer)
    # train(dataset_train, dataset_val , args)
    # fold 별

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    # train_idx, valid_idx 뱉어준다.
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train_dataset, train_dataset['label']), 1):
        if not args.cv:
            if fold > 1:
                break
        print(f'>> Cross Validation {fold} Starts!')
        # load dataset
        train_df = train_dataset.iloc[train_idx]
        valid_df = train_dataset.iloc[valid_idx]

        train_label = label_to_num(train_df['label'].values, args)
        valid_label = label_to_num(valid_df['label'].values, args)

        train(train_df, valid_df, train_label, valid_label, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training arguments
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=3, help='total number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size per device during training (default: 32)')
    parser.add_argument('--valid_batch_size', type=int, default=128,
                        help='batch size for evaluation (default: 128)')
    parser.add_argument('--model_name', type=str, default='klue/roberta-large',
                        help='what kinds of models (default: klue/roberta-large)')
    parser.add_argument('--cv', type=bool, default=False, help='using cross validation (default: False)')
    
    # training arguments that don't change well
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='output directory (default: ./results)')
    parser.add_argument('--save_total_limit', type=int, default=5, help='number of total save model        (default: 5)')
    parser.add_argument('--train_type', type=str, default='default',
                        help='default: (using 30 label) or '
                        'nop: (no_relation vs org vs per) or '
                        'org: (org details) or per: (per details)')
    parser.add_argument('--save_steps', type=int, default=100, help='model saving step')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning_rate (default: 5e-5)')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='number of warmup steps for learning rate scheduler (default: 100)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='strength of weight decay (default: 0.01)')
    parser.add_argument('--logging_dir', type=str, default='./logs',
                        help='directory for storing logs (default: ./logs)')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='log saving step (default: 100)')
    parser.add_argument('--evaluation_strategy', type=str, default='steps',
                        help='evaluation strategy to adopt during training (default: steps)')
    # `no`: No evaluation during training.
    # `steps`: Evaluate every `eval_steps`.
    # `epoch`: Evaluate every end of epoch.
    parser.add_argument('--eval_steps', type=int, default=100, help='evaluation step (default: 100)')
    parser.add_argument('--load_best_model_at_end', type=bool, default=True, help='(default: True)')
    parser.add_argument('--k_folds', type=int, default=5, help='number of cross validation folds (default: 5)')
    args = parser.parse_args()
    print(args)
    main(args)
