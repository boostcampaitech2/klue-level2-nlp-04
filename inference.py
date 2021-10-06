from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import DataLoader
from load_data import *
from model import CustomModel
from utils import *
import pandas as pd
import torch
import torch.nn.functional as F

import numpy as np
import argparse
from tqdm import tqdm


def inference(model, tokenized_sent, device, args):
    """
      test dataset을 DataLoader로 만들어 준 후,
      batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=args.batch_size, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        data = {k: v.to(device) for k, v in data.items() if k != 'labels'}
        with torch.no_grad():
            outputs = model(**data)
                # input_ids=data['input_ids'].to(device),
                # attention_mask=data['attention_mask'].to(device),
                # token_type_ids=data['token_type_ids'].to(device),
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()


def main(args):
    """
      주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    Tokenizer_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
    if args.tem:
        special_tokens_dict = {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>',
                                                             '<e3>', '</e3>', '<e4>', '</e4>']}
        tokenizer.add_special_tokens(special_tokens_dict)

    ## load test datset
    test_dataset_dir = "../dataset/test/test_data.csv"
    test_df = pd.read_csv(test_dataset_dir)
    test_id = test_df['id'].values.tolist()
    test_dataset = preprocessing_dataset(test_df, args)
    test_dataset, test_label = load_test_dataset(test_dataset, tokenizer, args)
    Re_test_dataset = RE_Dataset(test_dataset, test_label)

    ## load my model
    # args.run_name 으로 시작하는 model_folder 다 가져오기
    if args.inference_type == 'cv':
        model_list = glob(os.path.join(args.model_dir, args.model_name.split('/')[-1], args.run_name[0] + '*'))
        model_list = sorted(model_list)
    else:
        model_list = glob(os.path.join(args.model_dir, args.model_name.split('/')[-1], args.run_name[0]))

    print(model_list)

    output_probs = np.zeros((test_df.shape[0], 30))
    for model_name in model_list:
        MODEL_NAME = model_name
        if args.tem:
            model = CustomModel.from_pretrained(MODEL_NAME, model_name=MODEL_NAME)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        # model.parameters
        model.to(device)

        ## predict answer
        pred_answer, output_prob = inference(model, Re_test_dataset, device, args)  # model에서 class 추론

        output_probs += np.array(output_prob)

    output_prob = output_probs / len(model_list)
    output_prob = output_prob.tolist()
    pred_answer = np.argmax(output_probs, axis=-1).tolist()
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.
    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame({'id': test_id, 'pred_label': pred_answer, 'probs': output_prob, })

    output_dir = os.path.join(args.output_dir, args.inference_type, args.model_name.split('/')[-1])
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, f'{args.run_name[0]}_submission.csv')
    output.to_csv(output_dir, index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print('---- Finish! ----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument('--model_dir', type=str, default="/opt/ml/klue-level2-nlp-04/best_model/")
    parser.add_argument('--model_name', type=str, default='klue/roberta-large', help='what kinds of models')
    parser.add_argument('--inference_type', type=str, default="cv",
                        help='default: (using 30 label) or '
                             'cv: (exp_cv + exp_cv1 + exp_cv2 + exp_cv3 + exp_cv4) or ')
    parser.add_argument('--run_name', nargs='+', type=str, default='exp',
                        help='names of the W&B run inference_type default or cv: exp_cv or')
    parser.add_argument('--output_dir', type=str, default="./prediction")
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size per device during training (default: 512)')
    parser.add_argument('--tem', type=bool, default=True, help='using typed entity marker (default: "")')

    args = parser.parse_args()

    assert args.model_name, "사용할 model_name 을 적어주세요"
    assert args.run_name, "inference_type=default 사용할 run_name 을 적어주세요"
    assert args.tem, "typed entity marker 를 사용하신다면 True 아니라면 False 를 적어주세요"

    print(args)
    main(args)

