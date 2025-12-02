python 3.10.x 이 필요합니다.

### (선택) 가상환경 설정
다른 프로젝트와 패키지 버전이 섞이지 않도록,  
가상환경에서 실행하는 것을 권장합니다.

```bash
# 프로젝트 폴더로 이동
cd project_root

# 가상환경 생성 (Windows 기준)
python -m venv .venv

# 가상환경 활성화
.\.venv\Scripts\activate

# 필요 라이브러리 설치
pip install -r requirements.txt
```

### 0. 디렉터리 구조 조정
프로젝트 루트 폴더를 임의의 위치에 생성한 후에 다음과 같이 파일들이 위치하도록 조정해줍니다.
Food-11 파일을 그대로 data_raw 파일에 넣어주면 됩니다. 

최초 디렉터리 구조:
```
project_root/               # 프로젝트 루트 폴더
├─ prepare_food11_binary.py # Food-11 → diet/not_diet 데이터 재구성
├─ main.py                  # 모델 학습 / 평가 / 그래프 저장
├─ predict_single.py        # 학습된 모델로 이미지 1장 예측
└─ data_raw/                # 원본 Food-11 데이터
   └─ Food-11/
      ├─ training/
      │  ├─ Bread/
      │  ├─ Dairy product/
      │  └─ ... (총 11개 클래스)
      ├─ validation/
      └─ evaluation/
```

### 1. Dataset 준비
```bash
# prepare_food11_binary.py
python prepare_food11_binary.py
```

prepare_food11_binary.py 실행 후, 디렉터리 구조:
```
project_root/               # 프로젝트 루트 폴더
├─ prepare_food11_binary.py # Food-11 → diet/not_diet 데이터 재구성
├─ main.py                  # 모델 학습 / 평가 / 그래프 저장
├─ predict_single.py        # 학습된 모델로 이미지 1장 예측
├─ data_raw/                # 원본 Food-11 데이터
│  └─ Food-11/
│     ├─ training/
│     │  ├─ Bread/
│     │  ├─ Dairy product/
│     │  └─ ... (총 11개 클래스)
│     ├─ validation/
│     └─ evaluation/
└─ data/                    # 학습용으로 재구성된 이진 분류 데이터
   ├─ train/
   │  ├─ 0_not_diet/
   │  └─ 1_diet/
   ├─ val/
   │  ├─ 0_not_diet/
   │  └─ 1_diet/
   └─ test/
      ├─ 0_not_diet/
      └─ 1_diet/
```

### 2. 모델 학습 및 평가
```bash
# main.py
python main.py
```
   
### 3. 개별 이미지 파일 테스트
predict_single.py 내부에서 테스트하고자하는 이미지 파일의 경로를 변경해주어야 합니다.
```bash
# predict_single.py
python predict_single.py
```
