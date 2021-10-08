<div align="center">
  <h1>pstage_01_image_classification</h1>
</div>

![pstage](https://user-images.githubusercontent.com/51067412/132005999-5cb1ecf0-7992-4ee6-8788-567244d00dbf.PNG)

## :fire: Getting Started
### Dependencies
```python
torch==1.7.1
torchvision==0.8.2
tensorboard==2.6.0
pandas==1.1.5
opencv-python==4.5.1.48
scikit-learn==0.24.2
matplotlib==3.3.4
timm==0.4.12
tqdm==4.51.0
numpy==1.19.2
python-dotenv==0.19.0
Pillow==8.1.0
h5py==3.1.0
glob2==0.7
```

### Install Requirements
- `pip install -r requirements.txt`

### Contents
- `dataset.py`
- `face_image.py` : FaceNet 적용
- `loss.py` 
- `model.py` 
- `optimizer.py` 
- `train.py`
- `inference.py`
- `evaluation.py`
- `perfect_train.csv` : dataset

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`

### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`

## :mag: Overview
### Background
> COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 받고있습니다. </br>
> 확산을 막기위해 많은 노력들을 하고 있지만 COVID-19의 강한 전염력 때문에 우리를 오랫동안 괴롭히고 있습니다. </br>
> 이를 해결하는 방법은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. </br>
> 이를 위해 우리는 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하는 시스템이 필요합니다. </br>
> 즉, **카메라로 비춰진 사람 얼굴 이미지만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, </br>
> 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다.**

### Problem definition
> 카메라로 비춰진 사람 얼굴 이미지만으로 이 사람이 마스크를 쓰고 있는지, </br>
> 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템 or 모델

### Development environment
- GPU V100 원격 서버
- PyCharm 또는 Visual Studio Code | Python 3.7(or over)

### Evaluation
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F6390139%2Fb19f3db709b41788c3b1333ef1ae11a9%2Ff1score.png?generation=1608093256720406&alt=media">

## :mask: Dataset Preparation
### Prepare Images
<img width="1103" alt="스크린샷 2021-09-03 오후 11 05 14" src="https://user-images.githubusercontent.com/68593821/132018480-dcc7ddc0-e019-4797-8b98-72fa97b69856.png">
<h6>출처 : kr.freepik.com</h6>

- 전체 사람 수 : 4500명 (train : 2700 | eval : 1800)
- age : 20대 - 70대
- gender : 남,여
- mask : 개인별 정상 착용 5장, 비정상적 착용 1장(코스크,턱스크...), 미착용 1장
- 전체 31,500 Images (train : 18,900 | eval : 12,600)
- 이미지 크기 : (384,512)

### Data Labeling
- mask, gender, age 기준 18개의 클래스로 분류
<img src="https://user-images.githubusercontent.com/68593821/131881060-c6d16a84-1138-4a28-b273-418ea487548d.png" height="500"/>

### [Facenet](https://arxiv.org/pdf/1503.03832.pdf)
```
$ python face_image.py
```
 - [face_image.py](https://github.com/boostcampaitech2/image-classification-level1-06/blob/main/face_image.py)
  - FaceNet 이용 얼굴 인식 후 FaceCrop
  - FaceCrop 후 Resize를 거친 이미지 크기 : (280, 210)

### Generate CSV files
- `perfect_train.csv`

## :running: Training

### Train Models
- [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf)
  - EfficientNet_b2_pruned 
  - EfficientNet_b4
- [SEResNet(SE block + ResNet)](https://arxiv.org/pdf/1709.01507.pdf)
  - SEResNet26d_32x4d
- [NFNet](https://arxiv.org/pdf/2102.06171.pdf)
  - NFNet_l0
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

### Multilabel Classification
```
# gender, age, mask 각각 학습
$ python train.py --multi=True
```


## :thought_balloon: Inference

```
# 단일 model을 통해 inference 시
$ python inference.py \
  --model_name={kinds of models} \
  --model_dir={model_filepath} \
  --pth_name={model parameter name} \
  --output_name={output_filename} \
```

```
# cross_validation 을 사용해 나온 model 5개를 통해 inference 시
$ python inference.py \
  --model_name={kinds of models} \
  --model_dir={model_filepath} \
  --pth_name=best \
  --output_name={output_filename} \
  --cv=True
```

```
# ensemble 을 사용해 inference 시
$ python inference.py \
  --ensemble_model_name={kinds of models : 띄어쓰기로 구분한 여러개의 model_name} \
  --ensemble_model_dir={model_filepath : 띄어쓰기로 구분한 여러개의 model_file_path} \
  --pth_name=best \
  --output_name={output_filename} \
  --ensemble=True
```

```
# multilabel inference 시
$ python inference.py \
  --model_name={kinds of models} \
  --model_dir={model_filepath} \
  --pth_name=best \
  --output_name={output_filename} \
  -- multi True
```
## 📄 Wrap Up Report
[Wrap Up Report](https://github.com/boostcampaitech2/image-classification-level1-06/files/7109259/_level1_6.1.pdf)

