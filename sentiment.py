import streamlit as st
import joblib 
from deep_translator import GoogleTranslator


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

# 한국어 -> 영어 번역 함수
def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        return text

# 입력창 (안내 문구 변경)
user_input = st.text_area("리뷰 내용을 입력해주세요 (한국어, 영어 모두 가능합니다!)", height=120)

if st.button("감성 분석 실행"):
    if user_input.strip() == "":
        st.warning("분석할 내용을 입력해주세요!")
    else:
       with st.spinner("AI가 분석 중입니다..."):
            # 자동으로 영어로 번역 (한국어든 영어든 알아서 대응됨)
            processed_text = translate_to_english(user_input)
            
            # 벡터화 및 예측
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
                
         
                    