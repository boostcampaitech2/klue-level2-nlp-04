# 🏆 대회 개요
## Realtion Extraction
관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.

```
sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
subject_entity: 썬 마이크로시스템즈
object_entity: 오라클

relation: 단체:별칭 (org:alternate_names)
```
* input: sentence, subject_entity, object_entity의 정보를 입력으로 사용 합니다.
* output: relation 30개 중 하나를 예측한 pred_label, 그리고 30개 클래스 각각에 대해 예측한 확률 probs을 제출해야 합니다.

## Micro F1 score
micro-precision과 micro-recall의 조화 평균이며, 각 샘플에 동일한 importance를 부여해, 샘플이 많은 클래스에 더 많은 가중치를 부여합니다. 데이터 분포상 많은 부분을 차지하고 있는 no_relation class는 제외하고 F1 score가 계산 됩니다.
![image](https://user-images.githubusercontent.com/45033215/135595092-a81cb6f7-e333-42e2-acf2-a92e7b0be619.png)
![image](https://user-images.githubusercontent.com/45033215/135595162-6cfc2db0-e5bf-4dcb-b372-d5ed9f573c17.png)

## AURPC
x축은 Recall, y축은 Precision이며, 모든 class에 대한 평균적인 AUPRC로 계산해 score를 측정 합니다. imbalance한 데이터에 유용한 metric 입니다.

![image](https://user-images.githubusercontent.com/45033215/135595189-001353e5-bfe7-4bf5-9415-7a2a1543a082.png)


# 🚀 데이터 분석
## 데이터셋 통계
전체 데이터에 대한 통계는 다음과 같습니다.
* train.csv: 총 32470개
* test_data.csv: 총 7765개

## 데이터 예시
![image](https://user-images.githubusercontent.com/45033215/135595305-70c77c02-df9f-4796-a2d9-bad3ad901a8f.png)

## 30 Class 설명 (김상욱-T2033 캠퍼님 글 참고)

### [no_relation]
* 연관없음

---

### [organization]
* org:dissolved : 지정된 조직이 해산된 날짜
* org:founded : 지정된 조직이 설립된 날짜
* org:placeofheadqueaters : 지정된 조직의 본부가 있는 장소(본사위치)
* org:alternate_names : 지정된 조직을 참조하기 위해 사무실 이름 대신 호출되는 대체 이름
* org:member_of : 지정된 조직이 속한 조직
* org:members : 지정된 조직에 속한 조직
* org:political/religious_affiliation : 지정된 조직이 소속된 정치/종교 단체
* org:product : 특정 조직에서 생산한 제품 또는 상품
* org:founded_by : 특정 조직을 설립한 사람 또는 조직
* org:top_members/employees : 지정된 조직의 대표 또는 구성원
* org:numberofemployees/members : 지정된 조직에 소속된 총 구성원 수

---

### [person]
* per:dateofbirth : 지정된 사람이 태어난 날짜
* per:dateofdeath : 지정된 사람이 사망한 날짜
* per:placeofbirth : 특정인이 사망한 날짜
* per:placeofdeath : 특정인이 사망한 장소
* per:placeofresidence: 지정된 사람이 사는 곳
* per:origin : 특정인의 출신 또는 국적
* per:employee_of : 지정된 사람이 일하는 조직
* per:schools_attended : 지정된 사람이 다녔던 학교
* per:alternate_names : 지정된 사람을 지칭하기 위해 공식 이름 대신에 부르는 대체 이름
* per:parents : 지정된 사람의 부모
* per:children : 지정된 사람의 자녀
* per:siblings : 특정인의 형제자매
* per:spouse : 특정인의 배우자
* per:other_family : 부모, 자녀, 형제자매 및 배우자를 제외한 특정인의 가족
* per:colleagues : 지정된 사람과 함께 일하는 사람들
* per:product : 특정인이 제작한 제품 또는 작품
* per:religion : 특정인이 믿는 종교
* per:title : 특정인의 직위를 나타내는 공식 또는 비공식 이름


# 🌌 EDA
## 라이브러리
```py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
pd.options.display.max_colwidth = 300
```
* sentence columns를 모두 보기 위해 `pd.options.display.max_colwidth = 300` 를 설정했다.

## 데이터 프레임 확인
```py
df = pd.read_csv('./dataset/train/train.csv')
df.sample(5)
```
![image](https://user-images.githubusercontent.com/45033215/135596115-39517ab7-79a9-4700-8736-6feaefd89cc6.png)

```py
df.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32470 entries, 0 to 32469
Data columns (total 8 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   id              32470 non-null  int64 
 1   sentence        32470 non-null  object
 2   subject_entity  32470 non-null  object
 3   object_entity   32470 non-null  object
 4   label           32470 non-null  object
 5   source          32470 non-null  object
 6   sub_label       32470 non-null  object
 7   target          32470 non-null  int64 
dtypes: int64(2), object(6)
memory usage: 2.0+ MB
```

```py
df.label.unique(), df.label.nunique()
```
```
(array(['no_relation', 'org:member_of', 'org:top_members/employees',
        'org:alternate_names', 'per:date_of_birth',
        'org:place_of_headquarters', 'per:employee_of', 'per:origin',
        'per:title', 'org:members', 'per:schools_attended',
        'per:colleagues', 'per:alternate_names', 'per:spouse',
        'org:founded_by', 'org:political/religious_affiliation',
        'per:children', 'org:founded', 'org:number_of_employees/members',
        'per:place_of_birth', 'org:dissolved', 'per:parents',
        'per:religion', 'per:date_of_death', 'per:place_of_residence',
        'per:other_family', 'org:product', 'per:siblings', 'per:product',
        'per:place_of_death'], dtype=object),
 30)
```

## 클래스 분포 확인하기
```py
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.barh(sorted(df.label.unique(), reverse=True), df.label.value_counts().sort_index(ascending=False), color=sns.color_palette('hls', 30))
for idx, score in enumerate(df.label.value_counts().sort_index(ascending=False)):
    ax.text(s=score,x=score+50, y=idx, ha='left', va='center', fontweight='semibold')
    
plt.show()
```
![image](https://user-images.githubusercontent.com/45033215/135596485-1e4428cb-2712-4f82-a227-1958fdad631d.png)
* 데이터 불균형이 매우 심한 것을 알 수 있다.

---

각각의 sub label별로 확인해보자.
```py
df['sub_label'] = df.label.apply(lambda x : 'org' if 'org' in x else 'per' if 'per' in x else 'no_relation')

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.barh(sorted(df.sub_label.unique(), reverse=True), df.sub_label.value_counts().sort_index(ascending=False), color=sns.color_palette('hls', 3), height=0.5)
for idx, score in enumerate(df.sub_label.value_counts().sort_index(ascending=False)):
    ax.text(s=score,x=score+50, y=idx, ha='left', va='center', fontweight='semibold')
ax.set_xlim(0, 15000)
plt.show()
```
![image](https://user-images.githubusercontent.com/45033215/135606613-57b930b2-74fa-4c1f-acf7-c47851e8b6d4.png)

* 데이터의 비율이 균등하지는 않지만 sub label간 격차가 심하지 않음을 알 수 있다.

---

이번에는, Source에 따른 데이터를 확인해보자.
```py
df.source.unique()
```
```
array(['wikipedia', 'wikitree', 'policy_briefing'], dtype=object)
```
* source는 위와 같은 3개의 매체로 이루어져있다.


```py
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.barh(sorted(df.source.unique(), reverse=True), df.source.value_counts().sort_index(ascending=False), color=sns.color_palette('hls', 3))
for idx, score in enumerate(df.source.value_counts().sort_index(ascending=False)):
    ax.text(s=score,x=score+50, y=idx, ha='left', va='center', fontweight='semibold')
    
plt.show()
```
![image](https://user-images.githubusercontent.com/45033215/135608124-ad78bf65-83c6-4041-8dbf-87fc604c1a89.png)

* policycy_briefing 항목이 매우 작은 것은 알 수 있다.
* 추후에 Data Augmentation을 할 때는 성능이 잘 안나오는 데이터에 대해서도 해야하지만, 이렇게 데이터가 적은 데이터에 대해서도 해야겠다는 생각을 가지고 있어야 겠다.


```py
fig, axes = plt.subplots(3, 1, figsize=(5, 30), dpi=110)
for jdx, ax in enumerate(axes.flatten()):
    df_list = ["policy_briefing", "wikipedia", "wikitree"]
    df_ax = df[df['source'] == df_list[jdx]]
    label_count = len(df_ax.label.unique())
    ax.barh(sorted(df_ax.label.unique(), reverse=True), df_ax.label.value_counts().sort_index(ascending=False), color=sns.color_palette('hls', label_count))
    for idx, score in enumerate(df_ax.label.value_counts().sort_index(ascending=False)):
        ax.set_title(df_list[jdx], fontweight="semibold", fontsize=14)
        ax.text(s=score,x=score+(5 if jdx == 0 else 50 if jdx == 1 else 100), y=idx, ha='left', va='center', fontweight='normal')
        ax.set_xlim(0, 120 if jdx == 0 else 9000 if jdx == 1 else 4000)
    
plt.show()
```
![image](https://user-images.githubusercontent.com/45033215/135608291-505ca64b-0132-47cb-89c3-7eb92973c443.png)
![image](https://user-images.githubusercontent.com/45033215/135608357-55e36c5d-3836-4f18-94f7-0c8e403b2781.png)
![image](https://user-images.githubusercontent.com/45033215/135608373-4d7e0882-a669-4f6c-87e1-49af829bfd62.png)
* 각 source 별로 label의 분포를 볼 수 있다.
* wikipedia와는 달리 wikitree와 policy_briefing은 모든 라벨이 존재하지 않는다.



