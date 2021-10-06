import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
import tqdm

# utils.py: https://gist.github.com/Kitsunetic/833143d9cc89325c7e95bf3d3a0d4fcf <- seed_everything, make_result_dir 함수 원본
# MLM 원본 : https://dacon.io/competitions/official/235747/codeshare/3072#

################################################## util #################################################

def seed_everything(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

def make_result_dir(config):
    root_dir = Path(config.result_dir_root)
    # pp = root_dir / f"exp{config.exp_num}"
    pp = root_dir
    for i in range(999, -1, -1):
        ppf = pp / f"version_{i:03d}"
        if ppf.exists():
            break

    ppf = pp / f"version_{i+1:03d}"
    ppf.mkdir(parents=True, exist_ok=True)

    return ppf


@dataclass
class Config:
    seed: int = 8888
    result_dir_root = Path('/opt/ml/mlm') # ex) /opt/ml/mlm에 모델들을 저장할 생각이라면 '/opt/ml/mlm'을 넣어주면 됩니다.
    result_dir: Path = result_dir_root

    nlp_model_name: str = "klue/roberta-base"

    epochs: int = 2
    save_step: int = 30000

    dataset_dir: str = './' # ex) /opt/ml/dataset/train/train.csv 이라면 '/opt/ml/dataset/train' 부분을 넣어주면 됩니다(상대 경로도 가능합니다).
    # dataset_dir: str = '/opt/ml/dataset/test'
    # dataset_dir: str = '/opt/ml/dataset'
    batch_size: int = 32
    mlm_probability: float = 0.15

################################################## 데이터 전처리 + Dataset #################################################

'''
이 부분은 원본에 있던 코드이며 문장에서 한글과 영어만 filtering하고 학습시 해당 sentence만 줍니다. 이때 문자와 숫자가 붙어있으면 문자열이라고 생각하고 포함시킵니다.
ex) !@#$나는 !@#$ 밥을 1먹었다. -> 나는 밥을 1먹었다.
'''

def preprocessing(line):
    line = re.sub(r"[^가-힣a-zA-Z0-9\s]+", " ", line)
    line = " ".join(filter(lambda word: not word.isdigit(), line.split()))
    line = re.sub(r"\s{2,}", " ", line).strip()

    if len(line) >= 2 and line[1] == " ":
        line = line[2:].strip()
    return line

class DefaultDataset(Dataset):
    def __init__(self, dataset, tokenizer: PreTrainedTokenizer, config: Config, debug=False) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        self.df = dataset
        self.lines = []
        for _, row in self.df.iterrows():
            line = preprocessing(row.sentence)
            self.lines.append(line)

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int):
        line = self.lines[idx]
        token = self.tokenizer.encode_plus(line, max_length=self.tokenizer.model_max_length, truncation=True)
        return token

#--------------------------------------------------------------------------------------------------------------#

'''
이 부분은 baseline code에서 load_data.py와 동일합니다.
여기서는 학습을 위한 데이터셋을 subject_entity + [sep] + object_entity + dataset['sentence']형식으로 만듭니다.
'''

class RE_Dataset(Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, nums):
    self.pair_dataset = pair_dataset
    self.nums = nums

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    return item

  def __len__(self):
    return self.nums

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
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
      return_token_type_ids=False
      )
  return tokenized_sentences

def tokenized_dataset_tm(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
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
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False 
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
    tokenized_sentences['e3_mask'] = torch.tensor(e3_mask, dtype=torch.long)
    tokenized_sentences['e4_mask'] = torch.tensor(e4_mask, dtype=torch.long)


def main(config: Config):

################################################## 기본적인 셋팅 #################################################
    DATA_NAME = 'train_v1.csv'
    DATASET = config.dataset_dir + DATA_NAME # dataset 경로
    seed_everything(config.seed) # seed값 설정 

    model = AutoModelForMaskedLM.from_pretrained(config.nlp_model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(config.nlp_model_name)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=config.mlm_probability)

################################################ MLM용 data set 설정 #############################################

# 학습시 e01 + '[SEP]' + e02 + dataset['sentence'] 등의 데이터로 학습

    # dataset = load_data(DATASET)
    # # # train_dataset = tokenized_dataset(dataset,tokenizer)
    # train_dataset = tokenized_dataset_tm(dataset,tokenizer)
    # ds_train = RE_Dataset(pair_dataset=train_dataset, nums=len(dataset['id']))

#--------------------------------------------------------------------------------------------------------------#

# 학습시 dataset['sentence']만 가지고 학습

    ds_train = DefaultDataset(pd.read_csv(DATASET),tokenizer, config)

#--------------------------------------------------------------------------------------------------------------#

    # special_tokens_dict = {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>',
    #                                                          '<e3>', '</e3>', '<e4>', '</e4>']}
    # tokenizer.add_special_tokens(special_tokens_dict)

################################################## train 부분 ###################################################

    training_args = TrainingArguments(
        output_dir=config.result_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        save_steps=config.save_step,
        save_total_limit=1,
        logging_dir=config.result_dir / "log"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=ds_train,
    )

################################################ model save 부분 ##################################################

    trainer.train()
    trainer.save_model(config.result_dir) # 1번

    final_dir = config.result_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir) # 2번
    tokenizer.save_pretrained(final_dir)

    '''
     1번과 2번 둘 다 .from_pretrained 에서 사용할 수 있는 모델을 저장하는데 왜 굳이 2번이나 저장하는지는 모르겠습니다;;; 
     저는 1번 위치에 있는 모델을 사용해서 fine-tuning을 했습니다.
     + 2번을 사용하면 tokenizer 정보도 저장이 됩니다. -> special_tokens_map.json, tokenizer_config.json, tokenizer.json, vocab.txt 등
     
     reference 
     1번. save_model() : https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer
     2번. save_pretrained() : https://huggingface.co/transformers/main_classes/model.html
    '''


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    nlp_model_names = [
        "klue/roberta-large"
    ]
    for nlp_model_name in nlp_model_names:
        config = Config()
        config.nlp_model_name = nlp_model_name
        config.result_dir = make_result_dir(config)

        main(config)
