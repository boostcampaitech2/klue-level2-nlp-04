<div align="center">
  <h1>KLUE_Relation Extraction</h1>
</div>

![image](https://user-images.githubusercontent.com/85214907/136507555-d8d417b0-1c11-4ea5-bc73-739d28cf8c25.png)

## :fire: Getting Started
### Dependencies
```python
torch==1.7.1
transformers==4.10.0
pandas==1.1.5
scikit-learn==0.24.1
matplotlib==3.3.4
tqdm==4.51.0
numpy==1.19.2
glob2==0.7
```

### Install Requirements
- `pip install -r requirements.txt`

### Contents
- `hyperparameter_search.py`
- `inference.py`
- `load_data.py` 
- `loss.py` 
- `model.py` 
- `new_mlm.py` : MLM 적용
- `train.py`
- `utils.py`
- `train_v1.csv` : 40% of no_relation data

### Training
- `SM_CHANNEL_TRAIN=[train csv data dir] SM_MODEL_DIR=[model saving dir] python train.py`

### Inference
- `SM_CHANNEL_EVAL=[test csv dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`


## :mag: Overview
### Background
> 관계 추출(Relation Extraction)이란 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다.<br>
> 문장 속에서 단어간에 관계성을 파악하는 관계 추출(Relation Extraction)은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다. 


### Problem definition
> 주어진 문장과 문장의 단어(subject entity, object entity)를 이용하여, <br>
> subject entity와 object entity가 어떤 관계가 있는지 예측하는 시스템 or 모델 구축하기

### Development environment
- GPU V100 원격 서버
- PyCharm 또는 Visual Studio Code | Python 3.7(or over)

### Evaluation
![image](https://user-images.githubusercontent.com/22788924/136509413-3597fc20-6d08-4575-9927-745adc32bf65.png)

## Dataset Preparation
### Prepare Images
<img width="833" alt="스크린샷 2021-10-08 오후 3 27 35" src="https://user-images.githubusercontent.com/46557183/136508822-bc5a076c-b36a-4915-b6b5-693cae21938e.png">

- train.csv: 총 32470개
- test_data.csv: 총 7765개 (정답 라벨은 blind = 100으로 임의 표현)
- Input: 문장과 두 Entity의 위치(start_idx, end_idx)
- Target: 카테고리 30개 중 1개

<img width="955" alt="스크린샷 2021-10-08 오후 3 29 58" src="https://user-images.githubusercontent.com/46557183/136508992-7a8ff2bf-a5d9-4334-9cab-696a9c08f645.png">


### Data Labeling
- 크게 no-relation, org, per기준 30개의 클래스로 분류
<img src="https://user-images.githubusercontent.com/46557183/136508278-bad1fb56-23cd-4b2b-83d8-727bdebc06f3.png" height="500"/>

## :running: Training

### Train Models
- [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)
  - klue/roberta-small(https://huggingface.co/klue/roberta-small)
  - klue/roberta-base(https://huggingface.co/klue/roberta-base)
  - klue/roberta-large(https://huggingface.co/klue/roberta-large/tree/main)
- [BERT](https://arxiv.org/pdf/1810.04805.pdf)
  - klue/bert-base(https://huggingface.co/klue/bert-base)
- [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
- [koelectra-base](https://huggingface.co/monologg/koelectra-base-v3-discriminator)
```
# 단일 모델 train 시
$ python train.py 
```

### Stratified K-fold
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F3gQO8%2FbtqF0ZOHja8%2FSUTbGTYwVndcUJ5qWusqa0%2Fimg.png" height="250">


```py 
from sklearn.model_selection import StratifiedKFold
```
```
# cross_validation 사용해 train 시
$ python train.py --cv True
```
`train.py` cross_validation 함수



## :thought_balloon: Inference
```
# cross_validation 을 사용안할시
$ python inference.py \
  --model_name={kinds of models} \
  --model_dir={model_filepath} \
  --output_name={output_filename} \
  --inference_type=default \
  --run_name = exp\
  --cv = False\
  --tem = (typed entitiy 사용시 True, 아니면 False)
```

```
# cross_validation 을 사용해 나온 model 5개를 통해 inference 시
$ python inference.py \
  --model_name={kinds of models} \
  --model_dir={model_filepath} \
  --output_name={output_filename} \
  --inference_type = cv\
  --run_name = exp\
  --cv = True\
  --tem = (if typed entity: True, else: False)
```
