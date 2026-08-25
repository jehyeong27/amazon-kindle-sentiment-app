import streamlit as st
import joblib 
from googletrans import Translator

# 번역기 객체 생성
translator = Translator()

# 페이지 설정
st.set_page_config(page_title="감정 분석기")

st.title(" AI 감성 분석 서비스 (한국어/영어 지원)")
st.write("도서 리뷰를 입력하시면 언어에 상관없이 AI가 감성을 분석해 드립니다!")

@st.cache_resource
def load_model_and_vectorizer():
    loaded_model = joblib.load('model/sentiment_model.joblib')
    loaded_vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
    return loaded_model, loaded_vectorizer

model, vectorizer = load_model_and_vectorizer()

# 입력창 (안내 문구 변경)
user_input = st.text_area("리뷰 내용을 입력해주세요 (한국어, 영어 모두 가능합니다!)", height=120)

if st.button("감성 분석 실행"):
    if user_input.strip() == "":
        st.warning("분석할 내용을 입력해주세요!")
    else:
        with st.spinner("AI가 분석 중입니다..."):
            try:
                # 1. 입력된 텍스트가 한국어인지 감지하고 영어로 번역
                detected = translator.detect(user_input)
                if detected.lang != 'en':
                    translated = translator.translate(user_input, dest='en')
                    processed_text = translated.text
                else:
                    processed_text = user_input
                
                # 2. 텍스트 백터화 및 예측
                input_vector = vectorizer.transform([processed_text])
                prediction = model.predict(input_vector)[0]
                prediction_proba = model.predict_proba(input_vector)[0]
                
                # 3. 결과 출력
                st.subheader("분석 결과")
                if prediction == 1:
                    confidence = prediction_proba[1] * 100
                    st.success(f" 긍정적인 리뷰 입니다! (확신도: {confidence:.2f}%)")
                else:
                    confidence = prediction_proba[0] * 100
                    st.error(f" 부정적인 리뷰 입니다! (확신도: {confidence:.2f}%)")
                    
                # 번역된 텍스트
                if detected.lang != 'en':
                    st.caption(f"(참고 - 번역된 텍스트: {processed_text})")
                    
            except Exception as e:
                st.error(f"번역 및 분석 중 오류 발생: {e}")