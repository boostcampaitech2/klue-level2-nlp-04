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


def load_test_dataset(dataset_dir, tokenizer):
    """
      test dataset을 불러온 후,
      tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir)
    test_label = len(test_dataset)*[100]
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return test_dataset['id'], tokenized_test, test_label, test_dataset


def main(model_name, model_dir, analysis_dir):
    print(model_name, model_dir, analysis_dir)
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
    # model.parameters
    model.to(device)

    ## load test datset
    test_dataset_dir = "../dataset/train/stratified_dev.csv"
    test_id, test_dataset, test_label, test_df = load_test_dataset(test_dataset_dir, tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset, test_label)

    ## predict answer
    pred_answer, output_prob = inference(model, Re_test_dataset, device)  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

    output = pd.DataFrame({'id': test_id, 'pred_label': pred_answer, 'probs': output_prob, })
    test_df = pd.merge(test_df, output, how='outer', on='id')

    test_df.to_csv(analysis_dir+'.csv', index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    print('---- Finish! ----')


if __name__ == '__main__':
    experiment_list, model_list = get_experiment_dict()

    model_name, wandb_name = model_list[0]  # idx 0 is ("klue/roberta-base", "KLUE-RoBERTa-base")
    experiment_name = experiment_list[2]  # idx 2 is "DataAug-AEDA"
    # model_dir = os.path.join('./best_model/', experiment_name, wandb_name)
    checkpoint = 'checkpoint-3700'
    model_dir = os.path.join('./results', experiment_name, wandb_name, checkpoint)
    analysis_dir = os.path.join('./analysis/', f'{experiment_name}_{wandb_name}_{checkpoint}')

    ### 🔥🔥🔥🔥🔥🔥🔥🔥
    ### 실행전 반드시 아래를 확인할 것!
    ### model_name
    ### wandb_name
    ### experiment_name
    ### 🔥🔥🔥🔥🔥🔥🔥
    main(model_name=model_name,
         model_dir=model_dir,
         analysis_dir=analysis_dir,
         )
