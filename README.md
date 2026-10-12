# Amazon Kindle 리뷰 감성 분석

Amazon Kindle 도서 리뷰의 감성을 분석하는 Streamlit 앱입니다. 한국어 또는 영어 리뷰를 입력하면 영어로 번역을 시도한 뒤, 학습된 머신러닝 모델을 사용해 긍정 또는 부정으로 분류하고 예측 확률을 보여줍니다.

## 주요 기능

- 한국어 및 영어 리뷰 입력
- `deep-translator`를 이용한 영어 번역
- TF-IDF 벡터화와 학습된 분류 모델을 이용한 감성 예측
- 긍정/부정 결과 및 확신도 표시

## 실행 방법

Python 가상 환경을 만든 뒤 프로젝트 루트에서 다음 명령을 실행합니다.

```bash
python -m venv .venv
```

가상 환경을 활성화하고 의존성을 설치합니다.

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

앱을 실행합니다.

```bash
streamlit run sentiment.py
```

브라우저에서 Streamlit이 출력하는 로컬 주소를 열고 리뷰를 입력하세요.

## 저장소 구조

```text
.
├── model/
│   ├── sentiment_model.joblib
│   └── tfidf_vectorizer.joblib
├── requirements.txt
├── sentiment.ipynb
└── sentiment.py
```

- `sentiment.py`: Streamlit 감성 분석 앱
- `sentiment.ipynb`: 데이터 탐색 및 모델 작업용 노트북
- `model/`: 앱에서 사용하는 학습된 모델과 TF-IDF 벡터라이저
- `requirements.txt`: Python 패키지 목록

## 참고 사항

리뷰 번역에는 Google 번역 서비스가 사용되므로 번역 시 인터넷 연결이 필요합니다. 앱은 저장소 루트에서 실행해야 `model/` 경로의 파일을 찾을 수 있습니다.
