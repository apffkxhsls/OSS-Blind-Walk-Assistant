import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import streamlit as st
from PIL import Image

from config import TEST_IMAGES_DIR
from detection import BrailleDetector
from utils.tts import speak_guidance
from utils.alert import play_warning_sound, get_alert_message

st.set_page_config(page_title="BrailleGuard", page_icon="🦯", layout="wide")

@st.cache_resource
def load_detector():
    return BrailleDetector()

detector = load_detector()

st.title("🦯 BrailleGuard — 보행 방해물 실시간 탐지")
st.caption("시각장애인을 위한 YOLO 기반 인도 위 방해물 탐지 시스템")

mode = st.radio("입력 방식", ["📷 카메라", "🖼️ 이미지 업로드", "🗂️ 테스트 이미지"], horizontal=True)

image_np = None

if mode == "📷 카메라":
    cam = st.camera_input("카메라로 촬영")
    if cam:
        image_np = np.array(Image.open(cam).convert("RGB"))

elif mode == "🖼️ 이미지 업로드":
    uploaded = st.file_uploader("이미지 업로드", type=["jpg","jpeg","png"])
    if uploaded:
        image_np = np.array(Image.open(uploaded).convert("RGB"))

elif mode == "🗂️ 테스트 이미지":
    test_imgs = list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png"))
    if test_imgs:
        selected = st.selectbox("테스트 이미지 선택", [f.name for f in test_imgs])
        image_np = np.array(Image.open(TEST_IMAGES_DIR / selected).convert("RGB"))
    else:
        st.info(f"`{TEST_IMAGES_DIR}` 에 테스트 이미지를 추가하세요.")

if image_np is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본")
        st.image(image_np, use_column_width=True)

    with st.spinner("탐지 중..."):
        detections = detector.predict(image_np)
        result_img = detector.draw_boxes(image_np, detections)

    with col2:
        st.subheader("탐지 결과")
        st.image(result_img, use_column_width=True)

    st.divider()
    if detections:
        st.subheader(f"탐지된 객체 — {len(detections)}개")
        for det in detections:
            risk = "🔴 고위험" if det["is_high_risk"] else "🟡 주의"
            st.markdown(f"**{risk}** · {det['class_name']} · 신뢰도 {det['confidence']:.1%}")
        alert_msg = get_alert_message(detections)
        if alert_msg:
            st.error(f"⚠️ {alert_msg}")
            play_warning_sound()
            speak_guidance(alert_msg)
    else:
        st.success("✅ 방해물이 감지되지 않았습니다.")