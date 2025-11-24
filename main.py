import streamlit as st
import os
from dotenv import load_dotenv
import subprocess
import sys

# Load environment variables
load_dotenv()
app_password = os.getenv("APP_PASSWORD")

if not app_password:
    st.error("❌ APP_PASSWORD not found in .env file!")
    st.stop()

st.set_page_config(page_title="Secure Access", layout="centered")
st.title("🔐 Secure Login")

# Ask for password
password_input = st.text_input("Enter Password", type="password")

if st.button("Login"):
    if password_input == app_password:
        st.success("✅ Access granted! Launching app...")
        
        # Run the data_bot.py file
        python_path = sys.executable
        subprocess.Popen([python_path, "-m", "streamlit", "run", "data_bot.py"])
        st.info("The Data Analysis App is now opening in a new browser tab.")
    else:
        st.error("❌ Incorrect Password. Try again.")
