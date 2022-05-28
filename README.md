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
> MIT-BIH와 관련된 데이터는, train data 87,554개와 test data 18,118개로 구성되어 있다.  
> 상단의 그림과 같이 레이블(클래스)이 달려있는데,  
> 각 벡터의 차원(dimension, 이해하기 어렵다면 길이라고 생각)의 끝에 존재한다.  
> 각 벡터는 레이블 포함 183 차원의 시계열 데이터(time-series)로 구성되어 있다.  
> 레이블을 제외한 차원은 심전도의 파형(wave)에 대한 정보이다.  
> 각 레이블의 정보는 아래와 같다.   
> reference: [ECG Heartbeat Classification: A Deep Transferable Representation](https://ieeexplore.ieee.org/abstract/document/8419425?casa_token=eOgA0A3Y3ngAAAAA:3D7mV0mtBCoIOmHrnHeCuADPcATXi7SCM7juaQ4McrrWKJehT1mfQQzLUYy48tNFoZQDNh2GFKOe)  
  
![레이블](https://user-images.githubusercontent.com/98927470/170815989-23e8a9a3-9409-47bf-b871-3c09477242ad.PNG)  
  
## 모델 구현
### 미리보는 전체 구조 그림
![블록구조](https://user-images.githubusercontent.com/98927470/170817186-2dd9debc-336d-4bc2-98df-dd0851eebd5a.png)  
![계층구조](https://user-images.githubusercontent.com/98927470/170817723-391569db-17dc-46b6-925b-e8870933a4ee.png)  
### ResNet & ResNeXt  
  
> 기울기 소실을 완화하고자 ResNet의 Residual learning 기법을 사용하였다.  
> 앙상블 기법과 비슷하게 입력이, 혹은 이전 단의 출력이 여러 갈래로 나뉘어 다음 단으로 진입하도록  
> 네트워크를 구성하였다. 하지만 완전히 앙상블과 똑같다고 말할 수는 없다.  
> ResNeXt의 기법을 사용하였기 때문인데, 나뉘어진 입력, 혹은 이전 단의 출력들이  
> 독립적으로 훈련되는게 아니기 때문이다.  
  
