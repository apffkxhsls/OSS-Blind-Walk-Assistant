import streamlit as st
from gtts import gTTS
from pathlib import Path

TEMP_FILE = Path("temp_voice.mp3")

def speak_guidance(text: str):
    try:
        tts = gTTS(text=text, lang="ko")
        tts.save(str(TEMP_FILE))
        st.audio(str(TEMP_FILE), format="audio/mp3", autoplay=True)
    except Exception as e:
        st.error(f"음성 안내 오류: {e}")
    finally:
        if TEMP_FILE.exists():
            TEMP_FILE.unlink()