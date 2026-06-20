import base64
import streamlit as st
import streamlit.components.v1 as components
from gtts import gTTS
from pathlib import Path

from utils.alert import _get_audio_base64

TEMP_FILE = Path("temp_voice.mp3")

def speak_guidance(text: str):
    try:
        tts = gTTS(text=text, lang="ko")
        tts.save(str(TEMP_FILE))

        with open(TEMP_FILE, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        st.iframe(
            f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            </audio>
            """,
            height=0,
        )
    except Exception as e:
        st.error(f"음성 안내 오류: {e}")
    finally:
        if TEMP_FILE.exists():
            TEMP_FILE.unlink()