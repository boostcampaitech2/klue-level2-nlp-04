# Code Analysis
├── train.py  
│　　├── library import and os setting  
│　　├── Generating Validation Data  
│　　├── F1 Score  
│　　├── Auprc  
│　　├── Computing Metrics  
│　　├── Replacing category to number on label  
│　　├── Train    
│　　└── Main  
├── experiment_dict.py  
│　　└── Experiment Dictionary  
├── load_data.py  
│　　├── Library import and os setting   
│　　├── RE_Dataset   
│　　├── Preprocessing    
│　　├── Data Loading  
│　　└── Data Tokeninzing  
└── inference.py  
　 　├── Library import and os setting  
　 　├── Inference  
　 　├── Replacing number to category on label    
　 　├── Data Loading    
　 　└── Main  

## train.py [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/Code%20Analysis.md#code-analysis)
> Library import and os setting
```py
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
```

---

> Generating Validation Data
```py
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
```
* stratified 하게 구성하며 80:20의 비율로 생성한다.
* 성공하면 True, 실패하면 False와 Error Message를 반환한다.
* 본 코드에서는 valid라는 용어대신 dev라는 용어를 사용한다.

---

> F1 Score

```py
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
```
* 예측값과 실제 라벨을 비교해서 f1 score를 계산한다.
* categorial한 label을 numerical 하게 바꾸기 위해 index와 매칭한다.
* 0번 인덱스를 없애는 이유는 

---

> Auprc

```py
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
```
* auprc 면적을 구하기 위한 함수이다.

> Computing Metrics

```py
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
```
* f1, auprc, accuracy를 구해서 반환한다.

> Replacing category to number on label

```py
def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label
```
* pickle에는 label과 num의 key-value 쌍으로 dictionary가 저장되어있다.
* 둘 사이를 변환할 수 있도록 pickle을 가지고 있으며 이 함수에서 변환이 이루어진다.

> Train

```py
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
```
* 모델 이름을 설정한다. 이 모델 이름으로 pre trained 모델과 tokenizer 모델을 결정하게된다.
* `new_dev_dataset` 인자가 True라면 dev dataset을 생성한다. False라면 생성되어있다고 가정하고 진행한다.
* 데이터셋을 불러온다. 이 데이터는 학습을 위해 원하는 형태로 전처리되어 불러와진다.
* 이 데이터셋을 라벨과 데이터로 분리하고 이 데이터를 토크나이징한다.
* 이후, 토크나이징 된 데이터와 숫자화 된 라벨을 묶어 RE_Dataset으로 선언한다.

---

```py
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

    wandb.init(project=experiment_name,
               name=wandb_name,
               config=wandb_config,
               reinit=True,
               )
    
    wandb_config.epochs = epoch
    wandb_config.batch_size = train_bs
    wandb_config.model_name = model_name,
    
    # train model
    trainer.train()
    wandb.finish()
    model.save_pretrained(os.path.join('./best_model/', experiment_name, wandb_name))
```
* 주어진 인자들로 모델의 하이퍼 파라미터를 설정한다.
* wandb로 결과를 시각화 하도록 log를 저장한다.
* 최고의 성능이 나왔을 때의 모델을 저장한다.
  * top 5 성능의 모델도 저장한다. 이 때 checkpoint의 이름으로 저장한다.
  * f1 score로 저장된다고 실험적으로 추측한다.

---

> Main
```py
def main():
    experiment_list, model_list = get_experiment_dict()

    # model_name, wandb_name = model_list[3]
    experiment_name = experiment_list[1]
    # for idx, (a, b) in enumerate(model_list.values()):
    # size = len()
    for idx in [1]:
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
```
* train에서 설정해야 할 파라미터들을 main에서 전달한다.
* 이 때 최고의 성능을 내기 위해 여러 실험을 진행하게 되는데 이러한 정보를 담고있는 dictionary를 불러올 수 있다.
* 이 dictionary는 get_experiment_dict()함수로 불러오며, 이 함수는 experiment_dict.py에 존재한다.


## experiment_dict.py [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/Code%20Analysis.md#code-analysis)
> Experiment Dictionary

```py
def get_experiment_dict():
    experiment_list = {
        0: "", 1: "Model", 2: "Loss", 3: "Batch",
        4: "LR", 5: "Warmup-Steps", 6: "K-Fold",
        7: "Multi-Label",
    }
    model_list = {
        0: ("klue/roberta-base", "KLUE-RoBERTa-base"),
        1: ("klue/roberta-large", "KLUE-RoBERTa-large"),
        2: ("klue/roberta-small", "KLUE-RoBERTa-small"),
        3: ("klue/bert-base", "KLUE-BERT-base"),
        4: ("monologg/koelectra-base-v3-discriminator", "KOELECTRA-base")
    }

    return experiment_list, model_list
```
* 현재는 model의 종류를 바꿔가면서 하는 실험만 했다. experiment_list에 있는 것보다 더 많은 부분을 실험할 예정이다.


## load_data.py [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/Code%20Analysis.md#code-analysis)
> Library import and os setting
```py
import pickle as pickle
import os
import pandas as pd
import torch
```

---

> RE_Dataset
```py
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
```
* RE Task에 사용할 Dataset을 구성한다.
* 이 Dataset의 data는 토크나이징을 거친 후의 data이며, label은 수치화되었다.

---

> Preprocessing

```py
def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
        i = eval(i)['word']  # eval(): str -> dict
        j = eval(j)['word']

        subject_entity.append(i)
        object_entity.append(j)
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'], 'subject_entity':subject_entity, 'object_entity':object_entity, 'label':dataset['label'],})
    return out_dataset
```
* dataset은 id, sentence, subject_entity, object_entity, label, source의 6가지 컬럼으로 되어있다.
* 이 때 subject_entity와 object_entity는 또 word, start_idx, end_idx, type의 4가지 key로 구성된 딕셔너리로 구성되어있다.
* 이 중 entity에 해당하는 word를 불러와서 리스트에 넣고 그 외에 id, sentence, label과 함께 DataFrame 으로 생성한다.
* 이는 학습을 위해 필요한 형태로 제작하는 과정이다.

---

> Data Loading

```py
def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset
```
* dataset_dir에는 csv 파일의 경로가 적혀있으며 이를 불러와 preporcessing 함수를 거친뒤 반환된다.
* train과 dev를 위해 두번 사용된다.

---

> Data Tokeninzing

```py
def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
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
        return_token_type_ids=False,
    )
    return tokenized_sentences
```
* model_name을 가지고 불러온 pretrained tokenizer와 label과 분리된, 원하는 형태의 data를 받는다.
* subject_entity와 object_entity를 [SEP] 토큰으로 분리해서 concat_entity에 추가한다.
* 이후 sentence와 concat_entity를 tokenizer에 입력한다.
* 추후에 entity를 어떻게 tokenizing 할까에 대한 실험을 통해 이 부분을 자세히 설명한다.


## inference.py [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/Code%20Analysis.md#code-analysis)
> Library import and os setting

```py
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
from experiment_dict import get_experiment_dict
```

---

> Inference

```py
def inference(model, tokenized_sent, device):
    """
      test dataset을 DataLoader로 만들어 준 후,
      batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=512, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                # token_type_ids=data['token_type_ids'].to(device)
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()
```
* tokenizing을 마친 sentence와 model을 입력받는다.
* 이후 예측값과 실제값의 차이를 구하고 softmax와 argmax를 거쳐 값을 구한다.
* 예측 라벨은 output_pred에, 각각의 softmax 확률은 output_prob에 추가한다.
* 모든 test_data에 대한 예측이 끝나면 output_pred와 output_prob을 concat해서 반환한다.

---

> Replacing number to category on label 

```py
def num_to_label(label):
    """
      숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label
```
* 예측으로 얻은 결과를 범주형 결과로 변환해야 한다.

---

> Data Loading

```py
def load_test_dataset(dataset_dir, tokenizer):
    """
      test dataset을 불러온 후,
      tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir)
    test_label = list(map(int, test_dataset['label'].values))
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return test_dataset['id'], tokenized_test, test_label
```
* train처럼 지정된 path에서 csv를 읽어 data를 불러온다.
* train과 다른점은, 어떠한 형태로 변환 없이 바로 토크나이징한다는 점.
* 이후 id와 토크나이징 sentence 그리고 label을 반환한다.

---

> Main

```py
def main(model_name, model_dir, prediction_dir):
    """
      주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    Tokenizer_NAME = model_name
    # Tokenizer_NAME = "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    ## load my model
    MODEL_NAME = model_dir  # model dir.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.parameters
    model.to(device)

    ## load test datset
    test_dataset_dir = "../dataset/test/test_data.csv"
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset, test_label)

    ## predict answer
    pred_answer, output_prob = inference(model, Re_test_dataset, device)  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame({'id': test_id, 'pred_label': pred_answer, 'probs': output_prob, })

    output.to_csv(prediction_dir+'.csv',
                  index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print('---- Finish! ----')


if __name__ == '__main__':
    experiment_list, model_list = get_experiment_dict()

    model_name, wandb_name = model_list[0]
    experiment_name = experiment_list[1]

    main(model_name=model_name,
         model_dir=os.path.join('./best_model/', experiment_name, wandb_name),
         prediction_dir=os.path.join('./prediction/', experiment_name, wandb_name),
         )
```
* test data에 대한 inference를 하기위해 모델을 불러온다.
* 이 때 모델은 experiment_dict.py의 get_experiment_dict()를 통해 가져온다.
* test_data.csv를 불러와 tokenizing하고 RE dataset으로 선언한다.
* 실제 inference를 통해 예측 결과와, 30개의 label에 대한 확률을 얻는다.
* 얻은 예측 결과를 범주형 데이터로 변환한다.
* 이 결과를 csv파일로 생성해 저장한다.
