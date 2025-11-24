import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import json
import re
import plotly.express as px
from dotenv import load_dotenv

# =======================
# 🔧 CONFIGURATION
# =======================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY not found in environment variables (.env).")
    st.stop()

genai.configure(api_key=api_key)

# =======================
# ⚙️ STREAMLIT FRONTEND
# =======================
st.set_page_config(page_title="AI Data Analyzer", layout="wide")
st.title("🧠 AI Data Analyzer & Visualizer (Gemini + CSV + Plotly)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "df" not in st.session_state:
    st.session_state.df = None

# =======================
# 📂 CSV FILE UPLOAD
# =======================
uploaded_file = st.file_uploader("📁 Upload a CSV file to analyze", type=["csv"])

if uploaded_file is None:
    st.info("⬆️ Please upload a CSV file to start.")
    st.stop()

# Read CSV into DataFrame (needed for visualization & sampling)
try:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
except Exception as e:
    st.error(f"❌ Failed to read CSV file: {e}")
    st.stop()

st.success("✅ CSV loaded successfully!")
st.dataframe(st.session_state.df.head(10), use_container_width=True)

df = st.session_state.df  # shorthand

# =======================
# 💬 CHAT INPUT (shown just below upload)
# =======================
user_input = st.chat_input("Ask about your data or request a visualization...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    sample_data = df.head(20).to_dict(orient="records")

    # Detect if visualization is intended
    if any(word in user_input.lower() for word in ["plot", "graph", "chart", "visualize", "draw", "show"]):
        # Visualization prompt
        prompt = f"""
        You are a data visualization assistant.
        Based on this dataset sample:
        {sample_data}

        User query: {user_input}

        Respond STRICTLY in this JSON format only:

        {{
          "type": "bar" | "line" | "pie" | "histogram",
          "x": "column_name",
          "y": "column_name" (omit for pie),
          "color": "optional_column_name"
        }}
        """

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt).text.strip()

        # Extract JSON safely
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            response_json = json_match.group(0)
        else:
            response_json = "{}"

        try:
            viz_spec = json.loads(response_json)
            chart_type = viz_spec.get("type")
            x = viz_spec.get("x")
            y = viz_spec.get("y")
            color = viz_spec.get("color")

            fig = None
            if chart_type == "bar":
                fig = px.bar(df, x=x, y=y, color=color)
            elif chart_type == "line":
                fig = px.line(df, x=x, y=y, color=color)
            elif chart_type == "pie":
                fig = px.pie(df, names=x, values=y)
            elif chart_type == "histogram":
                fig = px.histogram(df, x=x, color=color)
            else:
                st.warning("⚠️ Unknown chart type received from LLM.")

            if fig:
                st.session_state.chat_history.append({"role": "plot", "content": fig})

        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"⚠️ Error creating visualization: {str(e)}"
            })

    else:
        # Normal text Q&A prompt
        prompt = f"""
        You are a data analyst assistant. Answer the user's question using the dataset provided.
        Dataset sample:
        {sample_data}

        User question: {user_input}

        Be concise, accurate, and only respond in plain text (no JSON).
        """

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt).text.strip()

        st.session_state.chat_history.append({"role": "assistant", "content": response})

# =======================
# 💬 DISPLAY CHAT
# =======================
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])
    elif message["role"] == "plot":
        st.plotly_chart(message["content"], use_container_width=True)
