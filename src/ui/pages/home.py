import cv2
import numpy as np
import streamlit as st
from PIL import Image
import time

from src.detection.detector import BrailleDetector
from config import CHECKPOINT_DIR, TEST_IMAGES_DIR
from src.utils.tts import speak_guidance
from src.utils.alert import play_warning_sound, get_alert_message

version = st.sidebar.selectbox(
    "모델 버전 선택",
    ["v2_best", "v3_best", "v4_best", "v5_best", "v6_best", "v7_best"]
)

@st.cache_resource
def load_detector(version: str):
    model_path = CHECKPOINT_DIR / f"{version}.pt"
    return BrailleDetector(model_path=model_path)

detector = load_detector(version)

def render_home_page():
    st.title("📡 실시간 보행 방해물 탐지")
    st.caption("웹캠을 통해 전방의 장애물을 실시간으로 탐지하고 음성 안내를 제공합니다.")

    mode = st.radio("입력 방식", ["📷 실시간 웹캠", "🖼️ 이미지 업로드", "🗂️ 테스트 이미지"], horizontal=True)

    if mode == "📷 실시간 웹캠":
        if "last_alert_time" not in st.session_state:
            st.session_state.last_alert_time = 0

        ALERT_COOLDOWN = 5  # 몇 초에 한 번 울릴지
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        alert_placeholder = st.empty()
        stop_button = st.button("탐지 중지")

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()

            if not ret:
                st.error("웹캠을 불러올 수 없습니다.")
                break

            detections = detector.predict(frame)
            frame = detector.draw_boxes(frame, detections)

            alert_msg = get_alert_message(detections)
            if alert_msg:
                now = time.time()
                if now - st.session_state.last_alert_time >= ALERT_COOLDOWN:
                    play_warning_sound()
                    speak_guidance(alert_msg)
                    st.session_state.last_alert_time = now
                alert_placeholder.error(f"⚠️ {alert_msg}")
            else:
                alert_placeholder.empty()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB", width=700)

        cap.release()
        cv2.destroyAllWindows()

    elif mode == "🖼️ 이미지 업로드":
        uploaded = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
        if uploaded:
            image_np = np.array(Image.open(uploaded).convert("RGB"))
            _show_result(image_np)

    elif mode == "🗂️ 테스트 이미지":
        test_imgs = list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png"))
        if test_imgs:
            selected = st.selectbox("테스트 이미지 선택", [f.name for f in test_imgs])
            image_np = np.array(Image.open(TEST_IMAGES_DIR / selected).convert("RGB"))
            _show_result(image_np)
        else:
            st.info(f"`{TEST_IMAGES_DIR}` 에 테스트 이미지를 추가하세요.")


def _show_result(image_np: np.ndarray):
    """이미지 업로드 / 테스트 이미지 공통 결과 표시"""
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본")
        st.image(image_np, use_container_width=True)

    with st.spinner("탐지 중..."):
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        detections = detector.predict(image_bgr)
        result_img = detector.draw_boxes(image_bgr, detections)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("탐지 결과")
        st.image(result_img, use_container_width=True)

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


if __name__ == "__main__":
    render_home_page()