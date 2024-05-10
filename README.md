![스크린샷 2024-04-06 221326](https://github.com/DS-21-DL-Project/Trash-classification/assets/156426539/fb92e807-64fe-4327-93f8-dc275b855faf)

---

## Introduction
<blockquote> &nbsp;제주의 클린 하우스 운영과 같이 쓰레기 분리 수거를 실시하는 지역에서는 </br> &nbsp;가정 내 생활 폐기물 분류 기준에 대한 인식 부족이 중요한 문제로 대두되고 있습니다. </br> &nbsp;한국 전역에서는 생활 폐기물 발생량이 증가하는 추세이며 분리 배출 오분류로 인한 경제적 손실 또한 증가하고 있습니다. </blockquote> </br>

### 이 프로젝트는 ...
#### &nbsp;&nbsp;&nbsp; Yolo의 고도의 정확성과 신속성을 활용해 생활 폐기물을 식별하고 분류함으로써 </br> 
#### &nbsp;&nbsp;&nbsp; 재활용 프로세스의 첫 단계인 재활용 가능 자원의 효율적 회수를 도모 환경 오염을 줄이는 데 기여하고자 합니다. </br>
#### &nbsp;&nbsp;&nbsp; 이를 통해 지구 환경을 보호하고 지속 가능한 발전을 이끌어내는 사회적 가치를 창출하고자 합니다. </br></br>

---

## Contents
- [1. 프로젝트 소개](#1-프로젝트-소개)
  * [배경](#배경)
  * [프로젝트 개요](#프로젝트-개요)
- [2. 생활 폐기물 데이터](#2-생활-폐기물-데이터)
  * [수집 과정](#수집-과정)
  * [kaggle 데이터](#kaggle-데이터)
  * [직접 수집 데이터](#직접-수집-데이터)
- [3. 데이터 전처리](#3-데이터-전처리)
- [4. 모델링](#4-모델링)
  * [모델링 절차](#모델링-절차)
  * [1차 시도](#1차-시도)
  * [4차 시도](#4차-시도)
  * [11차 시도](#11차-시도)
  * [최종 모델](#최종-모델)
- [5. 성능 평가](#5-성능-평가) 
- [6. 프로젝트 한계 및 과제](#6-프로젝트-한계-및-과제)
  * [한계점](#한계점)
  * [향후 과제](#향후-과제)

---

## 1. 프로젝트 소개
### 배경
- 제주에서는 가정의 올바른 생활 폐기물 분리 배출로 재활용되는 자원의 양을 높이기 위해 클린 하우스를 운영 중이지만 </br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;분리 배출 전 폐기물 분류 기준에 대해 모호한 경우 발생
- 생활 폐기물 발생량은 연도별 증가하고 있으며, 전국 평균 발생량 또한 증가 추세
- 분리 수거 오분류로 인한 경제적 손실은 2018년 한국환경공단 추정 약 4,000억 원에서 2022년 환경부 추정 약 8,000억원까지 증가
- 이에 분리 배출의 정확도 향상을 위해 생활 폐기물 데이터를 수집 및 분석해 객체를 탐지 및 분류시켜 </br> &nbsp; &nbsp; &nbsp; &nbsp; 가정의 분리 배출에 대한 기준 확립을 도와 지구 환경을 보호하고 지속 가능한 발전에 기여하고자 함 </br></br>


### 프로젝트 개요
1. 프로젝트명: **EcoSort Helper : Yolo를 활용한 스마트 분리 배출 도우미**
2. 수행자: 강수정, 강호진
3. 수행 기간: 1개월 (2024.3.04 \~ 4.03)
4. 목표: YOLO를 활용한 생활 폐기물 데이터 "객체 탐지 및 분류" </br>
&nbsp;![스크린샷 2024-04-03 213341](https://github.com/DS-21-DL-Project/Trash-classification/assets/156426539/0514fc61-0ea3-4032-b819-48a099c30d5f)


<h3  dir="auto"><a id="user-content--tech-stack-" class="anchor" aria-hidden="true" tabindex="-1" href="#-tech-stack-"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a> Tech Stack </h3>

&nbsp; &nbsp; Language & Library <p align="justify">&nbsp; &nbsp; &nbsp;<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"/> &nbsp; <img src="https://img.shields.io/badge/ultralytics-150458?style=for-the-badge&logo=ultralytics&logoColor=white"/> &nbsp; <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/> &nbsp; <img src="https://img.shields.io/badge/opencv-5AC710?style=for-the-badge&logo=opencv&logoColor=white"/> </br>
&nbsp; &nbsp; Other <p align="justify">&nbsp; &nbsp; &nbsp;<img src="https://img.shields.io/badge/Roboflow-A100FF?style=for-the-badge&logo=roboflow&logoColor=white"> &nbsp; <img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252"/> &nbsp; <img src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white"/> &nbsp; <img src="https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white"/>

---

## 2. 생활 폐기물 데이터
### 수집 과정
- kaggle에서 제공하는 'Garbage Image Dataset'를 기본 활용 </br> 
- 한국의 생활 폐기물에 대한 데이터가 부족하였기에 자택, 동네 공원, 쓰레기 처리장 등 직접 데이터 수집 

### kaggle 데이터
![002](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/d301fc49-6189-4c4f-ac40-e7b33971d3f3)
 
- 일반쓰레기&nbsp; |&nbsp; 플라스틱&nbsp; |&nbsp; 종이&nbsp; |&nbsp; 고철&nbsp; |&nbsp; 유리
  
### 직접 수집 데이터
![003](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/564e2e5e-b3c4-4e06-b5da-90f669b2323f)

- 스티로폼&nbsp; |&nbsp; 복합 쓰레기&nbsp; |&nbsp; 배경&nbsp; |&nbsp; 비닐류&nbsp; |&nbsp; 일반 쓰레기&nbsp; |&nbsp; 고철류(한국 버전)&nbsp; |&nbsp; 플라스틱(라벨 없는)

---
## 3. 데이터 전처리

✅ **클래스 지정**
</br>&nbsp; 환경부에서 고시하는 분리 배출 기준에 따라 생활 폐기물 데이터를 다음 7개의 기준으로 나눔 </br>
- 종이류
- 플라스틱류
- 캔*고철류
- 유리류
- 비닐류
- 스티로폼
- 일반쓰레기

</br>

✅ **라벨링 작업_Roboflow**

![image](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/586e8499-e37e-479d-81d7-54c8695861c2)

- 데이터 셋 생성 및 전처리 작업, 증강 작업

---

## 4. 모델링

### 모델링 절차

: 생활 폐기물 데이터를 YOLO v8 Nano에 여러차례 학습시키며 큰 영향 준 분기점 표시

![EcoSort Helper 프로젝트 _ 최종수정본_복사본-011](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/85fc34fc-5f33-4b5c-845e-60f277cbfccb)


</br></br>
### 1차 시도

![012](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/e900c0bb-6a89-4473-8a02-25bedf8b0575)


- YOLO Documnet의 기본 학습 방법, 증강 기법으로 학습 진행
- 특이 사항 : kaggle 데이터의 분포 불균형 (∵ 해외 기준 생활 폐기물 데이터이므로 국내 기준과 상이)

</br></br></br></br>

### 4차 시도

![013](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/5171bebd-b377-4bad-b972-b8b0287cde52)


- 클래스 불균형 해소 위한 추가 데이터 확보 후 표본 수 보완
- 학습 방법 : batch 24, imgsz 640 (∵ 소지하고 있는 노트북 하드웨어가 버틸 수 있는 최대 값) 

</br></br>

### 4차 시도가 분기점인 이유
![014](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/241ccaa3-ba67-4fec-bc67-57d659e887f6)

- 추가 데이터들 대부분이 복합 쓰레기(한 화면에 수많은 객체 존재)
- mAP50 지표 비교 : 1차 시도와 비교했을 시 4차 시도의 mAP50이 현저히 떨어짐

- 문제점 : 객체의 겹침, 복잡한 배경, 부분적 가려짐, 일관성 없는 조명과 그림자 등 존재
  => 많은 데이터 학습시키지 않는 이상 모델 정확도 떨어짐
- 보완 : 한 화면에 최대 10개 객체 존재한 복합 쓰레기로만 학습  

</br></br></br></br>

### 11차 시도

![015](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/20177a9b-6598-4578-9b8f-7aaebdcd5183)

- 추가 데이터 확보 및 하이퍼파라미터 추가 통한 정적인 사진 데이터에 대해 안정적 정확도 나타냄
- 특이 사항 : 동영상 데이터 모델 적용 (∵ 프로젝트 최종 목표인 사용자가 생활 폐기물 분류를 위해 실시간 카메라로 분류 응답을 내놓는 모델)

</br></br>

### 11차 시도 특이 사항
![016](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/a048c220-478d-4487-aaf2-d4acea87b954)

문제점
- 1 : 배경 데이터 생활 폐기물 객체로 인식
- 2 : 동영상 데이터를 라벨링 할 시 영상을 프레임 단위로 잘라 진행 => 객체가 흐려짐 => YOLO 정확도 떨어짐
- 3 : 일반 쓰레기 데이터의 다른 클래스의 비해 상대적으로 낮은 정확도(5~60%) 

보완 
- 1 : 대량의 배경 사진 데이터 추가 학습 
- 2 : 증강 기법 중 흐림 기능 사용해 추가 학습
- 3 : 대량의 일반 쓰레기 데이터 추가 _프로젝트 마감일까지

</br></br></br></br>

### 최종 모델
![017](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/b8e24ea0-9eb4-4096-8407-6d63a772cf03)

- 앞선 문제점들 최대한 보완 => 데이터 분포 초기 비교 시 균일
- 특이 사항 : 하이퍼파라미터 튜닝을 통한 최종 모델의 최적 값 파라미터 구함

![018](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/e1508a09-36e7-4988-a48c-f5c4960758ed)

- 소지한 하드웨어의 한계로 프로젝트 시간 상 추가 시도 어렵다 판단되어 최종 하이퍼파라미터 마무리


</br></br>

## 5. 성능 평가

### mAP50, mAP50-90

![019](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/58f2ae4c-6568-4a0a-b5b0-5b41bfdaac57)

- YOLO 모델 성능 평가 시 가장 중요한 지표인 mAP 50과 mAP 50-90 모두 최종 모델로 갈수록 개선

</br>

### Precision(정밀도), Recall(재현율)

![020](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/854a8d5c-b725-4706-bb7c-cdda086ed5f9)

- 정밀도와 재현율에 대해서도 마찬가지로 최종 모델로 갈수록 성능 개선

</br>

### 정확도

![021](https://github.com/DS-21-DL-Project/Trash-classification/assets/83691399/5071bd3b-0fec-41db-8eec-81d1d06d51a9)

- 이전 모델까지의 일반 쓰레기 데이터 정확도(5~60%)에 비해 최종 모델에서 정확도 향상

</br>

### 데모 영상 1
[![Video Label](http://img.youtube.com/vi/9rdbIPDRP-0/0.jpg)](https://youtu.be/9rdbIPDRP-0)

[![Video Label](http://img.youtube.com/vi/JA3J8qMjWiY/0.jpg)](https://youtu.be/JA3J8qMjWiY)

Roboflow에서 제공해주는 모델을 스마트폰에서 실행하여 테스트

### 데모 영상 2 _11차 시도 VS 최종 모델

[![Video Label](http://img.youtube.com/vi/CVucEqj2OW4/0.jpg)](https://youtu.be/CVucEqj2OW4)


</br></br>

## 6. 프로젝트 한계 및 과제
### 한계점
- 머신러닝 프로젝트와 달리 딥러닝 프로젝트를 진행하는데 모델 처리에 있어 소지하고 있는 하드웨어의 한계로 하이퍼파라미터를 돌리는데 역부족 
- 2명이서 프로젝트를 진행하다 보니 데이터 확보 및 라벨링 작업에 한계가 있어 원할한 데이터 확보 어려움
- 대량의 배경 데이터를 추가하였지만 가끔씩 생활 폐기물 데이터로 인식하는 버그 발생

### 향후 과제
- 국내 생활 폐기물 처리장의 동영상 데이터를 확보할 수 있다면 분리 수거 자동화 작업에도 적용 가능한 모델로 발전 가능성 
- YOLO에 대한 추가 공부 필요
