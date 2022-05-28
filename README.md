# ResNet을 변형하여 만든 모델로 ECG 데이터 분류
## 심전도 데이터(ElectroCardioGram, ECG)  
![ecg_1period](https://user-images.githubusercontent.com/98927470/170815322-c0cd92cd-84c1-400a-a077-90cf37ab205a.jpg)  
> reference: [링크](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=ddongssanbaj&logNo=220216005366)  
> 심전도는 심장 박동을 일으키는 전위를 기록한 그래프이다.  
> 본 모델에서는 학습 및 성능평가를 위해  
> MIT-BIH arrhythmia database(부정맥 데이터베이스)를 이용하였다.  
   
## MIT-BIH 부정맥 데이터베이스(MIT-BIH arrhythmia DB)
### 링크: [MIT-BIH arrhythmia DB in physionet.org](https://www.physionet.org/content/mitdb/1.0.0/)  
### DB 다운로드 링크: [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)  
>   
> MIT-BIH 부정맥 데이터베이스는 Beth Israel Deaconess Medical Center(hospital, BIH)와  
> MIT에서 1975년부터 1979년까지 시행한 연구에서부터 얻어낸 4,000개 이상의  
> 홀터 심전도 검사를 사용한 기록을 저장한 것이다.  
> 랜덤하게 실제 환자 23명의 기록을 선택하고, 주요 증상을 갖는 25명의 기록,  
> 총 48명의 기록으로 구성된 데이터베이스이다.  
  
## 데이터 구성  
![데이터구성](https://user-images.githubusercontent.com/98927470/170815473-22bf99ac-ca95-44bf-a327-460074fe2cd1.PNG)
>   
> DB 다운로드 링크를 통해 데이터셋을 다운로드 받으면,  
> MIT-BIH와 관련된 데이터는, train data 87,554개와 test data 18,118개로 구성되어 있다.  
> 상단의 그림과 같이 레이블(클래스)이 달려있는데,  
> 각 벡터의 차원(dimension, 이해하기 어렵다면 길이라고 생각)의 끝에 존재한다.  
> 각 벡터는 레이블 포함 183의 차원으로 구성되어 있다.  
> 레이블을 제외한 차원은 심전도의 파형(wave)에 대한 정보이다.  
> 각 레이블의 정보는 아래와 같다.  
>   
![레이블](https://user-images.githubusercontent.com/98927470/170815989-23e8a9a3-9409-47bf-b871-3c09477242ad.PNG)  
  
## 모델 구현
