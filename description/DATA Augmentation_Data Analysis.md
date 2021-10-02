# DATA Augmentation : Data Analysis

## 무의식의 질문
> 왜 Data Augmentation을 하는데 데이터 분석을 해야하는 것인가?
* 무분별한 Data Augmentation은 성능을 해칠 수도 있기 때문이다.

> 그러한 이유는?
* 우리가 가지고 있는 데이터가 Unbalanced 하기 때문이다. 만약 그렇지 않다면 전체적인 Augmentation은 도움이 될 것이다.
![image](https://user-images.githubusercontent.com/45033215/135720305-f6ff30cd-7e86-4277-ab07-70c17a543bd2.png)

> 어떠한 방법으로 Augmentation 할 것인가?
* 이에 대한 내용은 바로 뒤에서 자세히 이야기하겠지만 다음과 같은 3가지 방법의 경우 Data Augmentation을 할 것이며 이에 대한 모델의 성능을 실험적으로 비교해볼 것이다.
  * lower a number of class
  * more a number of incorrect label
  * more a number of perplex label

<br>

---

<br>

## Data Analysis
### 라이브러리
```py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
pd.options.display.max_colwidth = 300
```
### Dev dataset
20%의 비율로 나눈 Dev dataset인 stratified_dev.csv 파일을 KLUE-RoBERTa-base로 inference했다. 이 때 기존의 inference와 달리, 이전 DataFrame의 내용은 유지하고 새로 얻은 prediction과 probability를 concat해서 결과 csv 파일을 만들었다.
* 실제로는 `on=id`로 merge했다.

```py

```

KLUE/RoBERTa-base 모델을 선택한 이유는 다음과 같다.


KLUE/RoBERTa-base 모델로 stratified_dev.csv를 inference한 성능은 다음과 같다.



<br>

---

<br>

## Data Analysis : lower a number of class


<br>

---

<br>

## Data Analysis : more a number of incorrect label

<br>

---

<br>

## Data Analysis : more a number of perplex label


