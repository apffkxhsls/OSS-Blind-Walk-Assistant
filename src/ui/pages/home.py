import cv2
import streamlit as st

from src.detection.detector import BrailleDetector
from config import CHECKPOINT_DIR
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
    st.title("👁️ 실시간 보행 방해물 탐지")
    st.caption("웹캠을 통해 전방의 장애물을 실시간으로 탐지하고 음성 안내를 제공합니다.")

    # (노트북 기본 웹캠) 카메라
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    alert_placeholder = st.empty()

    # 스트림릿 화면에 정지 버튼 만들기
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
            play_warning_sound()
            speak_guidance(alert_msg)
            alert_placeholder.error(f"⚠️ {alert_msg}")
        else:
            alert_placeholder.empty()

        # OpenCV 컬러(BGR)를 스트림릿 컬러(RGB)로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame, channels="RGB", width=700)

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

# 페이지 테스트용 실행 코드
if __name__ == "__main__":
    render_home_page()