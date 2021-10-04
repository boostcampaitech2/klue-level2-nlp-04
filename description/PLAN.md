# 문장 내 개체간 관계 추출

|번호|계획|관련 자료|진행도|
|:---:|:------------:|:------------:|:------:|
|0|[EDA](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#0-eda-)|Dataset 알아보기|`100%`|
|1|[Code Analysis](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#1-code-analysis-)|코드에 대한 자세한 설명|`100%`|
|2|[Model Comparison](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#2-model-comparison-)|[KoElectra](https://github.com/monologg/KoELECTRA), [KLUE/RoBERTa](https://github.com/KLUE-benchmark/KLUE)|`100%`|
|3|[DATA Augmentation : Data Analysis](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#3-data-augmentation--data-analysis-)|||
|4|[DATA Augmentation : KoEDA](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#4-data-augmentation--koeda-)|https://github.com/toriving/KoEDA||
||Loss Function Experiment||
||Entity Special Token|[오피스아워 정리 2번](https://github.com/sangmandu/SangSangPlus/issues/101#issue-1011979770)||
||K-Fold Validation Training||

|5|[DATA Augmentation : Pororo](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#5-data-augmentation--pororo-) |https://github.com/kakaobrain/pororo||
||Hyperparameter Tuning|https://medium.com/distributed-computing-with-ray/hyperparameter-optimization-for-transformers-a-guide-c4e32c6c989b|
||DATA Augmentation : KoBart|[KoBart](https://github.com/SKT-AI/KoBART)||
||DATA Augmentation : KoGPT2|[KoGPT2](https://github.com/gyunggyung/KoGPT2-FineTuning)||

||Additional Pretraining on dataset by MASK token|https://dacon.io/competitions/official/235747/codeshare/3072||
||Fine-tuning like Pre-tranining|[오피스아워 정리 6번](https://github.com/sangmandu/SangSangPlus/issues/101#issue-1011979770) / [HuggingFace-Bert](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py)||
||Making Classifier Deeper||
||Mutli Label Classification||

||Hyperparameter Experiment|epoch, lr, batch_size, warmup_steps,||

|||||

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


## 2. Model Comparison [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#%EB%AC%B8%EC%9E%A5-%EB%82%B4-%EA%B0%9C%EC%B2%B4%EA%B0%84-%EA%B4%80%EA%B3%84-%EC%B6%94%EC%B6%9C)
#### 📌 진행 : 21년 10월 01일  
#### 📖 내용
* 
#### 🚀 [세부 사항](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/Model%20Comparison.md)

## 3. DATA Augmentation : Data Analysis [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#%EB%AC%B8%EC%9E%A5-%EB%82%B4-%EA%B0%9C%EC%B2%B4%EA%B0%84-%EA%B4%80%EA%B3%84-%EC%B6%94%EC%B6%9C)
#### 📌 진행 : 21년 10월 02일  
#### 📖 내용
* 
#### 🚀 [세부 사항](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/9e91385eb48c0fc797f519d9b3bf979968a265e3/description/DATA%20Augmentation_Data%20Analysis.md)

## 4. DATA Augmentation : KoEDA [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#%EB%AC%B8%EC%9E%A5-%EB%82%B4-%EA%B0%9C%EC%B2%B4%EA%B0%84-%EA%B4%80%EA%B3%84-%EC%B6%94%EC%B6%9C)
#### 📌 진행 : 21년 10월 02일  
#### 📖 내용
* 
#### 🚀 [세부 사항]()

## 5. DATA Augmentation : Pororo [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#%EB%AC%B8%EC%9E%A5-%EB%82%B4-%EA%B0%9C%EC%B2%B4%EA%B0%84-%EA%B4%80%EA%B3%84-%EC%B6%94%EC%B6%9C)
#### 📌 진행 : 21년 10월 02일  
#### 📖 내용
* 
#### 🚀 [세부 사항]()
