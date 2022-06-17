# noise_CIFAR-100
> #### Noising CIFAR-100 시험 1위

![image](https://user-images.githubusercontent.com/76248669/174354160-9bfc8a81-9ca3-4f2d-8e1c-1fc7a9f42c51.png)

모교 기계학습 기말고사를 [kaggle](https://www.kaggle.com/competitions/cifar-100-classification)을 통하여 진행하였습니다. 운좋게 1위를 기록하여서 제가 이번 학기동안 공부했던 것들을 여기에 정리하고자 합니다.

- 시험에서 진행되었던 noise 이미지는 제공할 수 없습니다. 

  **Version**

  - Python: 3.9.12 (main, Apr  4 2022, 05:22:27) [MSC v.1916 64 bit (AMD64)]
  - Numpy: 1.21.5
  - Matplotlib: 3.5.1
  - Pytorch: 1.4.1
  - sklearn: 1.0.2
  - OpenCV: 4.5.5

------

## CIFAR-100

주어진 데이터 설명은 다음과 같습니다.

딥러닝에서 많이 활용되는 이미지 데이터셋 (칼라 이미지)

- CIFAR-10과 CIFAR-100이 존재함 -뒤에 있는 숫자가 클래스의 개수를 가리킴 
- Train : 50,000개(48 x 48) - Test : 10,000개(48 x 48) 
- 사이즈 변경 : 36 x 36 에서 48 x 48 
- 노이즈 추가 : 랜덤 노이즈 추가

## Tuning

- Optimizer: NAdam lr: 3e-4, weight decay: 3e-5
- Scheduler: ReduceLROnPlateau
- Model: ResNet18, ResNet50, ResNet152

## Accuary

| Model     | Accuracy |
| --------- | -------- |
| ResNet18  | 62.5%    |
| ResNet50  | 62.3%    |
| ResNet152 | 74.3%    |

