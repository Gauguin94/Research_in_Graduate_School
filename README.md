# ResNet을 변형하여 만든 모델로 ECG 데이터 분류
## 심전도 데이터(ElectroCardioGram, ECG)  
![ecg_1period](https://user-images.githubusercontent.com/98927470/170816740-61509fff-935b-47d2-bdd8-71d68fdb8cd1.png)  
  
> 심전도는 심장박동의 주기 중에 일어나는 심장 근육의 전기적 활동 상태를  
> 그래프 상에 나타낸 것으로, P, QRS-complex, T파(wave)로 구성되어 있다.    
> 심전도 데이터에 관한 연구는 대부분 심전도 12유도를 통해 얻은 데이터를 사용한다.  
> 12유도는 표준사지유도, 단극사지유도, 흉부유도로 진행된다.  
> 본 모델에서는 학습 및 성능평가를 위해 MIT-BIH arrhythmia DB(부정맥 데이터베이스)를 이용하였다.  
   
## MIT-BIH 부정맥 데이터베이스(MIT-BIH arrhythmia DB)
### 데이터 관련: [MIT-BIH arrhythmia DB in physionet.org](https://www.physionet.org/content/mitdb/1.0.0/)  
### DB 다운로드: [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)  
>   
> MIT-BIH 부정맥 데이터베이스는 Beth Israel Deaconess Medical Center(hospital, BIH)와  
> MIT에서 1975년부터 1979년까지 시행한 연구에서부터 얻어낸 4,000개 이상의  
> 홀터 심전도 검사를 사용한 기록을 저장한 것이다.  
> 홀터 심전도 검사(Holter monitoring)는 표준사지유도 중 Ⅱ 유도와 흉부 유도를  
> 사용하여 측정하며 이를 통해 얻은 신호로 대부분의 심전도 분류나 식별 관련 연구가 진행된다.  
> 랜덤하게 실제 환자 23명의 기록을 선택하고, 주요 증상을 갖는 25명의 기록,  
> 총 48명의 기록으로 구성된 데이터베이스이다.  
  
## 데이터 구성  
![데이터구성](https://user-images.githubusercontent.com/98927470/170815473-22bf99ac-ca95-44bf-a327-460074fe2cd1.PNG)
>   
> DB 다운로드 링크를 통해 데이터셋을 다운로드 받으면,  
> MIT-BIH와 관련된 데이터는, train data 87,554개와 test data 21,892개로 구성되어 있다.  
> 상단의 그림과 같이 레이블(클래스)이 달려있는데,  
> 각 샘플은 레이블 포함 길이 183의 시계열 데이터(time-series)로 구성되어 있다.  
> 레이블은 각 샘플의 끝에 존재한다. (ex) ecg[182] => 레이블)  
> 레이블을 제외한 샘플 내 데이터는 심전도의 time step에 대응되는 값,  
> 즉, 파형(wave)에 대한 정보이다. (ex) ecg[:181])  
> 각 레이블의 정보는 아래와 같다.  
> reference: [ECG Heartbeat Classification: A Deep Transferable Representation](https://ieeexplore.ieee.org/abstract/document/8419425?casa_token=eOgA0A3Y3ngAAAAA:3D7mV0mtBCoIOmHrnHeCuADPcATXi7SCM7juaQ4McrrWKJehT1mfQQzLUYy48tNFoZQDNh2GFKOe)  
  
![레이블](https://user-images.githubusercontent.com/98927470/170815989-23e8a9a3-9409-47bf-b871-3c09477242ad.PNG)  
  
![데이터세부구성](https://user-images.githubusercontent.com/98927470/170824254-6f582ec4-446e-4df8-b99c-3ec411c36e5d.PNG)  
  
## 모델 구현
### 미리보는 전체 구조 그림  
------
![블록구조](https://user-images.githubusercontent.com/98927470/170817186-2dd9debc-336d-4bc2-98df-dd0851eebd5a.png)  
![계층구조](https://user-images.githubusercontent.com/98927470/170817723-391569db-17dc-46b6-925b-e8870933a4ee.png)  
  
  
### ResNet & ResNeXt  
------
> 기울기 소실을 완화하고자 ResNet의 Residual learning 기법을 사용하였다.  
> 앙상블 기법과 비슷하게 입력이, 혹은 이전 단의 출력이 여러 갈래로 나뉘어 다음 단으로 진입하도록  
> 네트워크를 구성하였다. 하지만 완전히 앙상블과 똑같다고 말할 수는 없다.  
> ResNeXt의 기법을 사용하였기 때문인데, 나뉘어진 입력 혹은 이전 단의 출력들이  
> 독립적으로 훈련되는게 아니기 때문이다.  
  
  
### Proposed connection  
------
![2차원텐서연산](https://user-images.githubusercontent.com/98927470/170822360-0387ee57-d925-462a-9804-6194108137ab.PNG)  
> 미리보는 전체 구조 그림에서, "proposed connection"이라고 표시된 부분이 보인다.  
> proposed connection을 이야기하기 전에 짚고 넘어가야할 것이 있다.  
> 보편적으로 사용되는 합성곱 연산(conv2d)에서는 위와 같은 기준을 갖고 연산이 진행된다.  
> 하지만 필자는 conv1d를 사용하였고, conv1d에서의 연산은 아래와 같이 진행된다.  
> 각 샘플에서의 time-step이 height에 해당한다!  
  
![1차원텐서연산](https://user-images.githubusercontent.com/98927470/170822407-6b822a64-a8d2-4c63-b6c5-6a766bd747c3.PNG)  
  
> proposed connection은 제일 처음으로 모델에 들어오는 입력을 표시된 각 부분에 전달하는 역할을 한다.  
> 각 부분에 전달된 값들은 그 부분에 존재하는, feature map과 concatenation 연산을 수행한다.  
> concatenation 연산을 사용한 유명 모델로는 DenseNet이 존재하는데,  
> DenseNet은 feature map 간 channel-wise concatenation 연산을 수행한다.  
> 구현된 모델은 concatenation 연산을 수행하는 부분은 동일하지만,  
> channel-wise가 아닌, 위 그림에서의 Height를 늘리는 결과의 concatenation을 수행한다.  
```python
X_ = [x1, x2] # if shape of small 'x' is (64, 128, 30) => (Batch size, Channel, Height)
X = torch.cat(X_, axis=2) # shape of capital 'X' is (64, 128, 60) => (Batch size, Channel, Height)
```
> 모델로 들어오기 이전인, 가장 초기의 입력값(x)과 feature map(h(x))들을 concatenation하여  
> 아래의 그림과 같은 연산을 수행하게 된다.  
> 이로써 초기 입력값의 패턴과 feature map들의 패턴의 조합에서  
> 새로운 패턴을 발견할 수 있는, 새로운 학습효과를 기대할 수 있다.  
> (Height에 맞춰 Weight 또한 늘어난다.)  
> 단, 그대로 전달하면 연산이 불가능하기 때문에  
> 연산이 가능하도록 tensor의 shape만 맞춰주는 작업을 수행한다.  
> 아래 그림은 위의 그림과 다르게 편의상 transpose된 tensor로 표현하였다.  
![새로운기법](https://user-images.githubusercontent.com/98927470/170823446-9c20e6c7-7e46-46d5-be4b-6b081d986316.png)  
  
## 결과
  
![결과](https://user-images.githubusercontent.com/98927470/170824330-3e595749-5860-4fd8-a294-dff7664b9997.PNG)
  
>  위 사진은 분류 정확도, 민감도(재현율), 정밀도,  
>  민감도와 정밀도의 조화평균, 그리고 Test data에 대한 Confusion Matrix를 나타낸 그림이다.  
>  MIT-BIH arrhythmia DB를 다룬 타 논문들과 비교해도 괜찮은 수치 혹은 훌륭한 수치라고 보여진다.   
>  앞선 데이터 구성에서 살펴봤듯이, 'S'와 'F'에 해당하는 샘플의 수가 매우 적어  
>  해당 부분에 대해서는 성능이 다소 낮음을 확인할 수 있다.  
>  하지만 세상의 모든 데이터는 이와 같은 불균형을 이루고 있다.  
>  이와 같은 데이터 불균형을 어느정도 완화하거나, 완벽히 해소하는 것이 앞으로의 과제일 것이다.  
  
# 코드  
[WARP](https://github.com/Gauguin94/Research_in_Graduate_School/tree/main/ecgAnalysis) <- click  
