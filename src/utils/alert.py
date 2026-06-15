import streamlit as st
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import WARNING_SOUND, HIGH_RISK_CLASSES


def play_warning_sound():
    if WARNING_SOUND.exists():
        st.audio(str(WARNING_SOUND), format="audio/mp3", autoplay=True)
    else:
        st.warning(f"경고음 파일 없음: {WARNING_SOUND}")


def get_alert_message(detections: list[dict]) -> str | None:
    high_risk = [d for d in detections if d["is_high_risk"]]
    if not high_risk:
        return None
    names = list({d["class_name"] for d in high_risk})
    if len(names) == 1:
        return f"전방에 {names[0]}이 감지되었습니다. 주의하세요."
    joined = ", ".join(names[:-1]) + f" 및 {names[-1]}"
    return f"전방에 {joined}이 감지되었습니다. 주의하세요."