from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
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
            # token_type_ids=data['token_type_ids'].to(device)
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

    ## load test datset
    test_dataset_dir = "../dataset/test/test_data.csv"
    test_df = pd.read_csv(test_dataset_dir)
    test_dataset = preprocessing_test_dataset(test_df)
    test_id, test_dataset, test_label = load_test_dataset(test_dataset, tokenizer, args)
    Re_test_dataset = RE_Dataset(test_dataset, test_label)

    if args.inference_type in ['default', 'cv']:
        ## load my model
        # args.run_name 으로 시작하는 model_folder 다 가져오기
        model_list = sorted(glob(os.path.join(args.model_dir, args.model_name.split('/')[-1], args.run_name[0] + '*')))
        output_probs = np.zeros((test_df.shape[0], 30))
        for model_name in model_list:
            MODEL_NAME = model_name
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

            # model.parameters
            model.to(device)

            ## predict answer
            pred_answer, output_prob = inference(model, Re_test_dataset, device, args)  # model에서 class 추론
            pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

            output_probs += np.array(output_prob)

        output_prob = output_probs / len(model_list)
        output_prob = output_prob.tolist()
        pred_answer = np.argmax(output_probs, axis=-1).tolist()
        pred_answer = num_to_label(pred_answer)
        ## make csv file with predicted answer
        #########################################################
        # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
        output = pd.DataFrame({'id': test_id, 'pred_label': pred_answer, 'probs': output_prob, })
    elif args.inference_type == 'hierarchical':
        ## load my model
        nop_model_dir = os.path.join(args.model_dir, args.model_name.split('/')[-1], 'nop')
        org_model_dir = os.path.join(args.model_dir, args.model_name.split('/')[-1], 'org')
        per_model_dir = os.path.join(args.model_dir, args.model_name.split('/')[-1], 'per')

        nop_model = AutoModelForSequenceClassification.from_pretrained(nop_model_dir).to(device)
        org_model = AutoModelForSequenceClassification.from_pretrained(org_model_dir).to(device)
        per_model = AutoModelForSequenceClassification.from_pretrained(per_model_dir).to(device)

        # nop_model 을 사용해 no_relation vs org vs per class 추론
        print('nop predict')
        pred_answer, output_prob = inference(nop_model, Re_test_dataset, device, args)
        pred_answer = num_to_label(pred_answer, 'nop')  # 숫자로 된 class를 원래 문자열 라벨로 변환.
        test_df['nop_label'] = pred_answer
        test_df['pred_label'] = pred_answer
        test_df['probs'] = torch.softmax(torch.from_numpy(np.random.rand(test_df.shape[0], 30)),
                                         axis=-1).numpy().tolist()

        # nop_model 에서 org(1) 로 분류된 애들가지고
        org_test_df = test_df[test_df.nop_label == 'org']
        test_dataset = preprocessing_test_dataset(org_test_df)
        test_id, test_dataset, test_label = load_test_dataset(test_dataset, tokenizer, args)
        Re_test_dataset = RE_Dataset(test_dataset, test_label)

        # org_model 을 사용해 org 세부 class 추론
        print('org predict')
        pred_answer, org_output_prob = inference(org_model, Re_test_dataset, device, args)
        pred_answer = num_to_label(pred_answer, 'org')  # 숫자로 된 class를 원래 문자열 라벨로 변환.
        test_df.loc[test_df.nop_label == 'org', 'pred_label'] = pred_answer
        # test_df.loc[test_df.nop_label == 'org', 'probs'] = output_prob

        # nop_model 에서 per(2) 로 분류된 애들가지고
        per_test_df = test_df[test_df.nop_label == 'per']
        test_dataset = preprocessing_test_dataset(per_test_df)
        test_id, test_dataset, test_label = load_test_dataset(test_dataset, tokenizer, args)
        Re_test_dataset = RE_Dataset(test_dataset, test_label)

        # per_model 을 사용해 per 세부 class 추론
        print('per predict')
        pred_answer, per_output_prob = inference(per_model, Re_test_dataset, device, args)
        pred_answer = num_to_label(pred_answer, 'per')  # 숫자로 된 class를 원래 문자열 라벨로 변환.
        test_df.loc[test_df.nop_label == 'per', 'pred_label'] = pred_answer
        # test_df.loc[test_df.nop_label == 'per', 'probs'] = output_prob

        output = test_df[['id', 'pred_label', 'probs']]
    else:
        raise ValueError('inference_type 을 default or cv or hierarchical 로 적어주세요.')

    output_dir = os.path.join(args.output_dir, args.inference_type, args.model_name.split('/')[-1])
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, f'{args.run_name[0]}_submission.csv')
    output.to_csv(output_dir, index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print('---- Finish! ----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument('--model_dir', type=str, default="./best_model")
    parser.add_argument('--model_name', type=str, default='klue/roberta-large',
                        help='what kinds of models (default: klue/roberta-large)')
    parser.add_argument('--inference_type', type=str, default="default",
                        help='default: (using 30 label) or '
                             'cv: (exp_cv + exp_cv1 + exp_cv2 + exp_cv3 + exp_cv4) or '
                             'hierarchical: (nop + org + per)')
    parser.add_argument('--run_name', nargs='+', type=str, default=[],
                        help='names of the W&B run '
                             'inference_type default or cv: exp_cv or'
                             'inference_type hierarchical: [nop org per]) 순서대로 적어주자.')
    parser.add_argument('--output_dir', type=str, default="./prediction")
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size per device during training (default: 512)')

    args = parser.parse_args()

    assert args.model_name, "사용할 model_name 을 적어주세요"
    assert args.run_name, "inference_type=default 사용할 run_name 을 적어주세요" \
                          "inference_type=hierarchical 사용할 run_name 을 nop org per 순으로 적어주세요"

    print(args)
    main(args)
