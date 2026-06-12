import streamlit as st
import cv2
from PIL import Image
import numpy as np

st.set_page_config(page_title="시각장애인 보행 보조 시스템", layout="wide")

st.title("👁️ 시각장애인 보행 방해물 실시간 탐지 시스템")
st.caption("SMWU 2026 OSS Final Team Project")

st.sidebar.header("설정 및 입력")
menu = st.sidebar.selectbox("기능 선택", ["이미지 업로드 탐지", "웹캠 실시간 탐지"])

if menu == "이미지 업로드 탐지":
    uploaded_file = st.file_uploader("방해물이 의심되는 인도 사진을 올려주세요.", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", use_container_width=True)
        st.info("YOLOv8 모델이 준비되면 여기에 탐지 결과가 표시됩니다.")

elif menu == "웹캠 실시간 탐지":
    st.warning("웹캠 연동 기능은 로컬 환경에서 테스트 가능합니다.")