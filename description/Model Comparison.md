# 1. 모델 개요
![image](https://user-images.githubusercontent.com/45033215/135703411-fe901bac-ed56-411f-932f-7ac89f4c3cd1.png)
* 훈련 배치 사이즈 : 32
* 검증 배치 사이즈 : 128
* 학습률 : 5e-5
* warmup_steps : 500
* saving_steps : 100
* validation ratio : 0.2
* split approach : stratified
* seed : 42
* 전체 epoch : 10
* 한 epoch 당 step : 810
* 각 모델 별 소요시간이 나타나 있다.

# 2. 모델 성능
주된 지표 4가지의 그래프는 다음과 같다.
![image](https://user-images.githubusercontent.com/45033215/135703460-2bbee4b3-71ec-4b3f-84fb-72b5603472ea.png)
* 대체적으로 DistillBERT가 성능이 안좋음을 알 수 있다.
* 대부분의 모델은 80전후의 성능을 가진다.
* 초반 상승폭이 큰 모델은 KLUE-RoBERTa-large와 KLUE-BERT-base이다.
* 전체적인 metrics는 증가하지만 loss 값은 2~3 epoch 사이에서 다시 증가한다.

4개의 그래프 중 F1 Score 그래프를 더 중점적으로 분석한다. 2 epoch 마다의 결과 그래프는 다음과 같다. 그래프는 클릭하면 더 자세히 볼 수 있다.
## Model Comparison on 2-epoch
### 2 epoch
![image](https://user-images.githubusercontent.com/45033215/135703526-cafa3249-ffd1-41ef-b029-41a9cc4a8c6c.png)
* 🥇 KLUE-BERT-base : 82.225
* 🥈 KLUE-RoBERTa-large : 81.746
* 🥉 KLUE-RoBERTa-base : 80.194

### 4 epoch
![image](https://user-images.githubusercontent.com/45033215/135703542-67cacf49-02ac-4f63-9592-789875ad068e.png)
* 🥇 KLUE-RoBERTa-large : 83.565
* 🥈 KLUE-RoBERTa-base : 83.075
* 🥉 KLUE-BERT-base : 82.702

### 6 epoch
![image](https://user-images.githubusercontent.com/45033215/135703546-d12c82be-40d7-46bc-bcc4-75e2cb42dcf5.png)
* 🥇 KLUE-RoBERTa-large : 83.365
* 🥈 KLUE-BERT-base : 82.971
* 🥉 KLUE-RoBERTa-base : 82.606

### 8 epoch
![image](https://user-images.githubusercontent.com/45033215/135703574-82ccde6f-53dc-4869-b480-c6a0395a1128.png)
* 🥇 KLUE-RoBERTa-large : 84.101
* 🥈 KoELECTRA-base : 83.02
* 🥉 KLUE-BERT-base : 83.016

### 10 epoch
![image](https://user-images.githubusercontent.com/45033215/135703675-7786cb36-f35a-4219-8b21-f6b5e639c52a.png)
* 🥇 KLUE-RoBERTa-large : 84.182
* 🥈 KLUE-BERT-base : 83.236
* 🥉 KoELECTRA-base : 83.067

### 결론
* 5 epoch 이전에는 RoBERTa-large와 BERT-base가 여러 step동안 성능 순위가 뒤바뀐다. 이 때 주로 RoBERTa-base 까지가 3등안에 위치한다.
* 5 epoch 이후에는 BERT-base와 KoELECTRA-base가 여러 step동안 성능 순위가 뒤바뀐다. 이 때 RoBERTa-large는 계속 1등의 성능을 유지한다.
* RoBERTa-large와 BERT-base 그리고 RoBERTa-base는 시작부터 제일 큰 상승폭을 그리며 10epoch동안 상위에 위치한다.
* KoELECTRA는 초반에는 전체 8등 중 6등의 성적까지 기록하면서 비교적 안좋은 성능을 보이지만, 7 epoch 이후에는 2~4등을 바꿔가며 위치한다.
* mBERT는 모든 epoch동안 상위 3개의 모델과 지속적으로 1~2점의 차이를 가진다. multilingual model이지만 아무리 그래도 한국어 데이터셋으로만 pretrained 된 모델과의 성능 차이는 어쩔 수 없다는 결론.
* XLM-RoBERTa-base와 DistilBERT-base는 시작부터 끝까지 7, 8등의 성적을 가진다. 아무래도 한국어 데이터셋으로 pretrained 되지 않아서 이러한 결과가 발생했을 가능성이 높다.

# 리더보드 모델 제출 결과
다음과 같은 6개의 모델을 제출했다. 모델 선정 기준은 초반 epoch와 후반 epoch에서 각각 성능이 제일 좋았던(실제로 checkpoint로도 남아있었던) 체크포인트를 선택했다. 
|모델 - checkpoint|F1 Score|Auprc|
|:----------:|:---:|:---:|
|KLUE-RoBERTa-large-checkpoint-2400|61.667|69.615|
|KLUE-RoBERTa-large-checkpoint-8100|65.157|65.231|
|KLUE-RoBERTa-base-checkpoint-1600|58.887|63.065|
|KLUE-RoBERTa-base-checkpoint-8100|63.065|60.843|
|KLUE-BERT-base-checkpoint-1600|65.767|67.799|
|KLUE-BERT-base-checkpoint-8100|64.032|63.378|

### 결론
* RoBERTa의 경우 epoch가 어느 정도 있어야 학습이 잘 된다.
* 반면, BERT의 경우 적은 epoch에서 학습이 제일 잘 되는 것으로 보인다.
* 80%의 데이터셋을 학습하는 것보다 100%의 데이터셋을 학습하는 것이 성능이 더 좋다.
  * 추후에 K-Fold로 개선해야한다.
### 추측
* RoBERTa의 경우 minimum point가 10 epoch 이후에 존재할 수도 있다.
  * 희락님 실험 결과 20 epoch 에서는 성능이 오히려 감소되었다.
* Data Augmentation을 거치면 RoBERTa-large에서 높은 성능을 기록할 수 있을 것으로 보인다.
  * 모델이 깊은데 그만큼 데이터도 풍부해졌기 때문.
  * 그러나 이 부분은 실험적으로 해봐야 안다.
*  모델 사용에 있어서 `RoBERTa-large` `RoBERTa-base` `BERT-base` `KoELECTRA-base` 의 4가지 모델로 실험하는 것이 가장 좋아보인다.
