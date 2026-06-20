import cv2
import streamlit as st

from src.detection.detector import BrailleDetector
from src.utils.tts import speak_guidance
from src.utils.alert import play_warning_sound, get_alert_message

@st.cache_resource
def load_detector():
    return BrailleDetector()

def render_home_page():
    st.title("👁️ 실시간 보행 방해물 탐지")
    st.caption("웹캠을 통해 전방의 장애물을 실시간으로 탐지하고 음성 안내를 제공합니다.")

    detector = load_detector()

    # (노트북 기본 웹캠) 카메라
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    alert_placeholder = st.empty()

    # 스트림릿 화면에 정지 버튼 만들기
    stop_button = st.button("탐지 중지")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()

        mock_detections = [
            {"class_name": "킥보드", "is_high_risk": True},
            {"class_name": "자동차", "is_high_risk": True}]

        alert_msg = get_alert_message(mock_detections)

        if alert_msg: 
            play_warning_sound()     # 띵동! 소리 재생
            speak_guidance(alert_msg) # "전방에 킥보드 및 자동차이 감지되었습니다..." TTS 안내
            
            st.error(alert_msg) # 화면에도 빨간 창으로 경고 띄우기

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

        # OpenCV 컬러(BGR)를 스트림릿 컬러(RGB)로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame, channels="RGB", width=700)

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

# 페이지 테스트용 실행 코드
if __name__ == "__main__":
    render_home_page()