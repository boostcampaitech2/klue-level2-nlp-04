import pickle as pickle
import os
import pandas as pd
import torch
import argparse
import wandb
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
from experiment_dict import get_experiment_dict

os.environ['WANDB_SILENT']="true"

def generate_dev(seed):
    try:
        df = pd.read_csv("../dataset/train/train.csv")
        train_df, dev_df = train_test_split(df, test_size=0.2,
                                            stratify=df.label,
                                            random_state=seed,)
        train_df.to_csv("../dataset/train/stratified_train.csv")
        dev_df.to_csv("../dataset/train/stratified_dev.csv")

    except Exception as e:
        return False, e
    return True, ''


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
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc' : auprc,
        'accuracy': acc,
    }


def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def train(model_name, experiment_name, new_dev_dataset, train_dataset_path, dev_dataset_path,
          seed, epoch, train_bs, dev_bs, lr, warmup_steps, wandb_name):

    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    if new_dev_dataset:
        success, e = generate_dev(seed)
        if not success:
            print("e")
            raise Exception

    train_dataset = load_data(train_dataset_path)
    dev_dataset = load_data(dev_dataset_path) # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #print(device)
    # setting model hyperparameter
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    #print(model.config)
    model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=os.path.join('./results/', wandb_name),       # output directory
        save_total_limit=5,              # number of total save model.
        save_strategy="steps",          # save interval : "steps", "epoch", "no"
        save_steps=100,                 # model saving step.
        num_train_epochs=epoch,              # total number of training epochs
        learning_rate=lr,               # learning_rate
        per_device_train_batch_size=train_bs,  # batch size per device during training
        per_device_eval_batch_size=dev_bs,   # batch size for evaluation
        warmup_steps=warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=100,            # evaluation step.
        load_best_model_at_end=True,
        seed=seed,
        report_to="wandb",
    )
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
    )

    # wandb setting
    wandb_config = wandb.config
    wandb_config.epochs = epoch
    wandb_config.batch_size = train_bs
    wandb_config.model_name = model_name,

    wandb.init(project=experiment_name,
               name=wandb_name,
               config=wandb_config,
               reinit=True,
               )

    # train model
    trainer.train()
    wandb.finish()
    model.save_pretrained(os.path.join('./best_model/', experiment_name, wandb_name))


def main():
    experiment_list, model_list = get_experiment_dict()

    # model_name, wandb_name = model_list[3]
    experiment_name = experiment_list[1]
    # for idx, (a, b) in enumerate(model_list.values()):
    # size = len()
    for idx in [4, 1]:
        print(list(model_list.values()))
        a, b = list(model_list.values())[idx]
        model_name, wandb_name = a, b

        train(
            model_name=model_name,
            experiment_name=experiment_name,
            new_dev_dataset=False,
            train_dataset_path="../dataset/train/stratified_train.csv",
            dev_dataset_path="../dataset/train/stratified_dev.csv",
            seed=42,
            epoch=10,
            train_bs=32,
            dev_bs=128,
            lr=5e-5,
            warmup_steps=500,
            wandb_name=wandb_name,
        )


if __name__ == '__main__':
    main()
