# 문장 내 개체간 관계 추출

|번호|계획|세부 설명|진행도|
|:---:|:------------:|:------------:|:------:|
|0|[EDA](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#0-eda-)|Dataset 알아보기|`100%`|
|1|[Code Analysis](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#1-code-analysis-)|코드에 대한 자세한 설명|`100%`|
|2|[Base Model Performance Evaluation](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#2-base-model-performance-evaluation-)|Ko-Robeta-Base 사용해서 베이스 성능 측정하기|완료. 정리 작성하기|
|3|[Model Comparison](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#3-model-comparison-)|mBERT, KoBERT, GPTBERT, (추가적으로 PPT 확인하기)|현재 비교중 3개 모델 완료|
||Hyperparameter Tuning||
||DATA Augmentation : KoEDA|https://github.com/toriving/KoEDA||
||DATA Augmentation : Pororo|https://github.com/kakaobrain/pororo||
||Entity Special Token|[오피스아워 정리 2번](https://github.com/sangmandu/SangSangPlus/issues/101#issue-1011979770)||
||Additional Pretraining on dataset by MASK token|https://dacon.io/competitions/official/235747/codeshare/3072||
||Fine-tuning like Pre-tranining|[오피스아워 정리 6번](https://github.com/sangmandu/SangSangPlus/issues/101#issue-1011979770)||
||Making Classifier Deeper||
||Mutli Label Classification||
||K-Fold Validation Training||
||Hyperparameter Experiment|epoch, lr, batch_size, warmup_steps,||
||Loss Function Experiment||
||||

다음과 같은 요소를 반드시 포함하여 비교할 것
* 베이스 모델 성능
* 해당 기능만을 추가했을 때의 성능
* 지금까지의 모든 테크닉을 적용한 성능

## 0. EDA [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#%EB%AC%B8%EC%9E%A5-%EB%82%B4-%EA%B0%9C%EC%B2%B4%EA%B0%84-%EA%B4%80%EA%B3%84-%EC%B6%94%EC%B6%9C)
#### 📌 진행 : 21년 09월 30일 ~ 10월 01일  
#### 📖 내용
* 대회 목적과 세부 데이터셋에 대해 소개한다.
* 데이터셋 내부를 분석하고 표와 그래프를 통해 시각화한다.
#### 🚀 [세부 사항](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/EDA.md)


## 1. Code Analysis [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#%EB%AC%B8%EC%9E%A5-%EB%82%B4-%EA%B0%9C%EC%B2%B4%EA%B0%84-%EA%B4%80%EA%B3%84-%EC%B6%94%EC%B6%9C)
#### 📌 진행 : 21년 09월 30일 ~ 10월 01일  
#### 📖 내용
* 3개의 baseline code를 간단히 분석한다.
* 1개의 필요에 의해 만든 code를 간단히 분석한다.
* 각 코드의 목적을 최소한으로 설명한다.
#### 🚀 [세부 사항](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/Code%20Analysis.md)

## 2. Base Model Performance Evaluation [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#%EB%AC%B8%EC%9E%A5-%EB%82%B4-%EA%B0%9C%EC%B2%B4%EA%B0%84-%EA%B4%80%EA%B3%84-%EC%B6%94%EC%B6%9C)
#### 📌 진행 : 21년 10월 01일 ~ 10월 02일  
#### 📖 내용
* 
#### 🚀 [세부 사항](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/Base%20Model%20Performance%20Evaluation.md)

## 3. Model Comparison [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#%EB%AC%B8%EC%9E%A5-%EB%82%B4-%EA%B0%9C%EC%B2%B4%EA%B0%84-%EA%B4%80%EA%B3%84-%EC%B6%94%EC%B6%9C)
#### 📌 진행 : 21년 10월 01일  
#### 📖 내용
* 
#### 🚀 [세부 사항](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/Model%20Comparison.md)

## 5.  [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#%EB%AC%B8%EC%9E%A5-%EB%82%B4-%EA%B0%9C%EC%B2%B4%EA%B0%84-%EA%B4%80%EA%B3%84-%EC%B6%94%EC%B6%9C)
#### 📌 진행 : 21년 10월 02일  
#### 📖 내용
* 
#### 🚀 [세부 사항]()

