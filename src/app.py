import streamlit as st
from ui.pages.home import render_home_page

st.set_page_config(page_title="BrailleGuard", page_icon="🦯", layout="wide")

render_home_page()