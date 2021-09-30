import argparse
import pickle as pickle
import os
import pandas as pd
import torch
import wandb
import random
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *


# 학습한 모델을 재생산하기 위해 seed를 고정
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

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


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
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
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
    }


def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def train():
    # load model and tokenizer
    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    train_dataset = load_data("../dataset/train/train.csv")
    train_dataset, valid_dataset = train_test_split(train_dataset,
                                                    test_size=0.2,
                                                    stratify=train_dataset['label'],
                                                    random_state=42)

    train_label = label_to_num(train_dataset['label'].values)
    valid_label = label_to_num(valid_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, args)
    tokenized_valid = tokenized_dataset(valid_dataset, tokenizer, args)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    print(model.config)
    # model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=f'{args.output_dir}/{args.run_name}',  # output directory
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
        compute_metrics=compute_metrics  # define metrics function
    )

    # wandb setting
    wandb_config = wandb.config
    wandb_config.seed = args.seed
    wandb_config.epochs = args.epochs
    wandb_config.batch_size = args.batch_size
    wandb_config.model_name = args.model_name,

    wandb.init(project=args.project_name,
               name=args.run_name,
               config=wandb_config,
               )

    # train model
    trainer.train()
    wandb.finish()
    model.save_pretrained(f'./best_model/{args.model_name}_{args.run_name}')


def main():
    seed_everything(42)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training arguments
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='total number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=30,
                        help='batch size per device during training (default: 30)')
    parser.add_argument('--valid_batch_size', type=int, default=128,
                        help='batch size for evaluation (default: 128)')
    parser.add_argument('--model_name', type=str, default='klue/roberta-large',
                        help='what kinds of models (default: klue/roberta-large)')
    parser.add_argument('--run_name', type=str, default='exp', help='name of the W&B run (default: exp)')

    # training arguments that don't change well
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='output directory (default: ./results)')
    parser.add_argument('--save_total_limit', type=int, default=5, help='number of total save model (default: 5)')
    parser.add_argument('--save_steps', type=int, default=500, help='model saving step')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning_rate (default: 5e-5)')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='number of warmup steps for learning rate scheduler (default: 500)')
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
    parser.add_argument('--eval_steps', type=int, default=500, help='evaluation step (default: 500)')
    parser.add_argument('--load_best_model_at_end', type=bool, default=True, help='(default: True)')
    parser.add_argument('--project_name', type=str, default='p_stage_klue',
                        help='wandb project name (default: p_stage_klue')


    args = parser.parse_args()
    print(args)

    main()
