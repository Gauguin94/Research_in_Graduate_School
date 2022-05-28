# ResNet을 변형하여 만든 모델로 ECG 데이터 분류하기.
## 심전도 데이터(ElectroCardioGram, ECG)  
>  
> 심전도는 심장 박동을 일으키는 전위를 기록한 그래프이다.  
> 본 모델에서는 학습 및 성능평가를 위해  
> MIT-BIH arrhythmia database(부정맥 데이터베이스)를 이용하였다.  
   
## MIT-BIH arrhythmia database
### 링크: [MIT-BIH arrhythmia database in physionet.org](https://www.physionet.org/content/mitdb/1.0.0/)  
### DB 다운로드 링크: [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)  
>   
> MIT-BIH 부정맥 데이터베이스는 Beth Israel Deaconess Medical Center(hospital, BIH)와  
> MIT에서 1975년부터 1979년까지 시행한 연구에서부터 얻어낸 4,000개 이상의  
> 홀터 심전도 검사를 사용한 기록을 저장한 것이다.  
> 랜덤하게 실제 환자 23명의 기록을 선택하고, 주요 증상을 갖는 25명의 기록,  
> 총 48명의 기록으로 구성된 데이터베이스이다.  
