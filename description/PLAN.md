# 문장 내 개체간 관계 추출

|번호|계획|세부 설명|진행도|
|:---:|:------------:|:------------:|:------:|
|0|[EDA](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#0-eda-)|Dataset 알아보기|`100%`|
|1|[Code Analysis](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#1-code-analysis-)|코드에 대한 자세한 설명||
|2|[Baseline Code Completion](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#2-baseline-code-completion-)|Validation csv 추가, seed 설정|거의 작성|
|3|[Base Model Performance Evaluation](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#3-base-model-performance-evaluation-)|Ko-Robeta-Base 사용해서 베이스 성능 측정하기|완료. 정리 작성하기|
|4|[Model Comparison](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#4-model-comparison-)|mBERT, KoBERT, GPTBERT, (추가적으로 PPT 확인하기)|현재 비교중 3개 모델 완료|
|5|Hyperparameter Tuning||
|6|DATA Augmentation : KoEDA|https://github.com/toriving/KoEDA||
|7|DATA Augmentation : Pororo|https://github.com/kakaobrain/pororo||
|8|Entity Special Token|[오피스아워 정리 2번](https://github.com/sangmandu/SangSangPlus/issues/101#issue-1011979770)||
|9|Additional Pretraining on dataset by MASK token||
|10|Fine-tuning like Pre-tranining|[오피스아워 정리 6번](https://github.com/sangmandu/SangSangPlus/issues/101#issue-1011979770)||
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
#### 📌 진행 : 21년 09월 30일  
#### 📖 내용
* 
#### 🚀 [세부 사항](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/EDA.md)


## 1. Code Analysis [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#%EB%AC%B8%EC%9E%A5-%EB%82%B4-%EA%B0%9C%EC%B2%B4%EA%B0%84-%EA%B4%80%EA%B3%84-%EC%B6%94%EC%B6%9C)
#### 📌 진행 : 21년 09월 30일  
#### 📖 내용
* 
#### 🚀 [세부 사항](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/Code%20Analysis.md)

## 2. Baseline Code Completion [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#%EB%AC%B8%EC%9E%A5-%EB%82%B4-%EA%B0%9C%EC%B2%B4%EA%B0%84-%EA%B4%80%EA%B3%84-%EC%B6%94%EC%B6%9C)
#### 📌 진행 : 21년 10월 01일  
#### 📖 내용
* 
#### 🚀 [세부 사항](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/e4dfd1f6aea9b1263d8eeaad7d3bee1eef280a82/Baseline%20Code%20Completion.md)

## 3. Base Model Performance Evaluation [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#%EB%AC%B8%EC%9E%A5-%EB%82%B4-%EA%B0%9C%EC%B2%B4%EA%B0%84-%EA%B4%80%EA%B3%84-%EC%B6%94%EC%B6%9C)
#### 📌 진행 : 21년 10월 01일  
#### 📖 내용
* 
#### 🚀 [세부 사항](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/Base%20Model%20Performance%20Evaluation.md)

## 4. Model Comparison [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#%EB%AC%B8%EC%9E%A5-%EB%82%B4-%EA%B0%9C%EC%B2%B4%EA%B0%84-%EA%B4%80%EA%B3%84-%EC%B6%94%EC%B6%9C)
#### 📌 진행 : 21년 10월 01일  
#### 📖 내용
* 
#### 🚀 [세부 사항](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/Model%20Comparison.md)

## 5. Model Comparison [⬆](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/PLAN.md#%EB%AC%B8%EC%9E%A5-%EB%82%B4-%EA%B0%9C%EC%B2%B4%EA%B0%84-%EA%B4%80%EA%B3%84-%EC%B6%94%EC%B6%9C)
#### 📌 진행 : 21년 10월 02일  
#### 📖 내용
* 
#### 🚀 [세부 사항](https://github.com/boostcampaitech2/klue-level2-nlp-04/blob/JSM/description/Model%20Comparison.md)

