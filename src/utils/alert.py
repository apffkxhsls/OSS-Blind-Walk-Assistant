import streamlit as st
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import WARNING_SOUND, HIGH_RISK_CLASSES

import time
import streamlit.components.v1 as components

CLASS_KO = {
    "bicycle": "자전거",
    "bollard": "볼라드",
    "car": "자동차",
    "damaged_braille_block": "점자블록 손상",
    "kickboard": "킥보드",
    "motorcycle": "오토바이",
    "trash": "쓰레기통",
    "utility_pole": "전신주",
}

def play_warning_sound():
    if WARNING_SOUND.exists():
        components.html(
            f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{_get_audio_base64()}" type="audio/mp3">
            </audio>
            """,
            height=0,
        )
    else:
        st.warning(f"경고음 파일 없음: {WARNING_SOUND}")

def _get_audio_base64() -> str:
    import base64
    with open(WARNING_SOUND, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_alert_message(detections):
    high_risk = [d for d in detections if d["is_high_risk"]]
    if not high_risk:
        return None
    names = list({CLASS_KO.get(d["class_name"], d["class_name"]) for d in high_risk})
    if len(names) == 1:
        return f"전방에 {names[0]}가 감지되었습니다. 주의하세요."
    joined = ", ".join(names[:-1]) + f" 및 {names[-1]}"
    return f"전방에 {joined}가 감지되었습니다. 주의하세요."