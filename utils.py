import pickle
import random
import re
from glob import glob
from pathlib import Path

import numpy as np
import sklearn
import torch

from sklearn.metrics import accuracy_score


# 학습한 모델을 재생산하기 위해 seed를 고정
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# 경로
def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


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


def num_to_label(label, flag='default'):
    """
      숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    if flag == 'default':
        with open('dict_num_to_label.pkl', 'rb') as f:
            dict_num_to_label = pickle.load(f)
    elif flag == 'nop':
        dict_num_to_label = {0: 'no_relation',
                             1: 'org',
                             2: 'per'}
    elif flag == 'org':
        dict_num_to_label = {0: 'org:number_of_employees/members',
                             1: 'org:dissolved',
                             2: 'org:political/religious_affiliation',
                             3: 'org:founded_by',
                             4: 'org:product',
                             5: 'org:members',
                             6: 'org:founded',
                             7: 'org:place_of_headquarters',
                             8: 'org:alternate_names',
                             9: 'org:member_of',
                             10: 'org:top_members/employees'}
    elif flag == 'per':
        dict_num_to_label = {0: 'per:place_of_death',
                             1: 'per:schools_attended',
                             2: 'per:religion',
                             3: 'per:siblings',
                             4: 'per:product',
                             5: 'per:place_of_birth',
                             6: 'per:other_family',
                             7: 'per:place_of_residence',
                             8: 'per:children',
                             9: 'per:date_of_death',
                             10: 'per:parents',
                             11: 'per:colleagues',
                             12: 'per:spouse',
                             13: 'per:alternate_names',
                             14: 'per:date_of_birth',
                             15: 'per:origin',
                             16: 'per:title',
                             17: 'per:employee_of'}
    else:
        raise Exception('flag 에 default, nop, org, per 를 적절하게 넣어주세요!')
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label
