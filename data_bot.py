# data_bot.py
import streamlit as st
import os
import pandas as pd
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai
import plotly.express as px

# ------------------ CONFIG & ENV ------------------ #
load_dotenv()
APP_PASSWORD = os.getenv("APP_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Immediate env checks before rendering anything else
if APP_PASSWORD is None:
    st.error("APP_PASSWORD missing in .env — add APP_PASSWORD=your_password and redeploy.")
    st.stop()

if GEMINI_API_KEY is None:
    st.error("GEMINI_API_KEY missing in .env — add GEMINI_API_KEY=your_key and redeploy.")
    st.stop()

# Configure Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini client: {e}")
    st.stop()

# ------------------ PASSWORD CHECK ------------------ #
def password_entered():
    """Callback for password input on_change"""
    st.session_state["password_correct"] = (st.session_state.get("password") == APP_PASSWORD)

def check_password():
    """
    Render password input and manage session state.
    Returns True only when correct password has been entered.
    Uses the same pattern as the example file you provided.
    """
    if "password_correct" not in st.session_state:
        # First time: ask for password (on_change will set password_correct)
        st.text_input("🔑 Enter Password:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state.get("password_correct"):
        # Wrong password: show box again and error
        st.text_input("🔑 Enter Password:", type="password", on_change=password_entered, key="password")
        st.error("❌ Incorrect password")
        return False
    else:
        return True

# ------------------ APP MAIN ------------------ #
def run_data_bot():
    st.set_page_config(page_title="AI Data Analyzer", layout="wide")
    st.title("🧠 AI Data Analyzer & Visualizer (Gemini + CSV + Plotly)")

    # Initialize session state containers
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "df" not in st.session_state:
        st.session_state.df = None
    if "uploaded_csv_text" not in st.session_state:
        st.session_state.uploaded_csv_text = None

    st.markdown("Upload a CSV file and then ask questions or request visualizations about the data.")

    # ------------------ CSV UPLOAD ------------------ #
    uploaded_file = st.file_uploader("📁 Upload a CSV file to analyze", type=["csv"])

    if uploaded_file is None:
        st.info("⬆️ Please upload a CSV file to start.")
        return

    # Save uploaded CSV text (we will send to Gemini and also keep a dataframe for visualization)
    try:
        # Read raw bytes and decode (fallback to latin-1)
        raw_bytes = uploaded_file.getvalue()
        try:
            csv_text = raw_bytes.decode("utf-8")
        except Exception:
            csv_text = raw_bytes.decode("latin-1")
        st.session_state.uploaded_csv_text = csv_text
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        return

    # Show a small preview and a button to send the CSV to Gemini for a quick summary
    st.success("✅ CSV uploaded.")
    col1, col2 = st.columns([3, 1])
    with col1:
        # Try to create a dataframe for visualization (if it's malformed, show error but still allow sending raw)
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not parse CSV into DataFrame: {e}")
            st.session_state.df = None

    with col2:
        if st.button("Send CSV to Gemini (get schema + sample)"):
            with st.spinner("Sending CSV to Gemini..."):
                try:
                    # Send truncated CSV text to Gemini (avoid sending extremely large files in prompt)
                    safe_text = st.session_state.uploaded_csv_text[:20000]  # first 20k chars
                    prompt = f"""
You are a helpful assistant. Provide a concise schema summary (column names and inferred types)
and show the first 5 rows for the CSV content provided between CSV START and CSV END.
Respond in plain text.

CSV START
{safe_text}
CSV END
"""
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    resp = model.generate_content(prompt)
                    gemini_reply = resp.text.strip()
                    st.text_area("Gemini response (schema + sample)", value=gemini_reply, height=300)
                except Exception as e:
                    st.error(f"Failed to call Gemini: {e}")

    # ------------------ CHAT & VISUALIZATION ------------------ #
    st.markdown("---")
    st.subheader("Chat with your data / Request visualizations")

    user_input = st.chat_input("Ask about your data or request a visualization...")

    if user_input:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Build a small sample for context (if df exists use it, otherwise create a tiny sample from raw CSV)
        if st.session_state.df is not None:
            sample_data = st.session_state.df.head(20).to_dict(orient="records")
        else:
            # Minimal fallback: send first few lines of raw CSV
            lines = (st.session_state.uploaded_csv_text or "").splitlines()[:10]
            sample_data = {"raw_preview_lines": lines}

        # If visualization intent detected, ask Gemini to return a JSON spec for plotting
        visualization_keywords = ["plot", "graph", "chart", "visualize", "draw", "show"]
        if any(word in user_input.lower() for word in visualization_keywords):
            prompt = f"""
You are a data visualization assistant.
Based on this dataset sample:
{sample_data}

User query: {user_input}

Respond STRICTLY in this JSON format only (no extra text):

{{ 
  "type": "bar" | "line" | "pie" | "histogram",
  "x": "column_name",
  "y": "column_name" (omit for pie),
  "color": "optional_column_name"
}}
"""
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(prompt).text.strip()

                # Extract JSON safely from response
                json_match = re.search(r"\{[\s\S]*\}", response)
                response_json = json_match.group(0) if json_match else "{}"

                viz_spec = json.loads(response_json)
                chart_type = viz_spec.get("type")
                x = viz_spec.get("x")
                y = viz_spec.get("y")
                color = viz_spec.get("color")

                fig = None
                if st.session_state.df is None:
                    st.warning("No parsed DataFrame available for plotting.")
                else:
                    if chart_type == "bar":
                        fig = px.bar(st.session_state.df, x=x, y=y, color=color)
                    elif chart_type == "line":
                        fig = px.line(st.session_state.df, x=x, y=y, color=color)
                    elif chart_type == "pie":
                        fig = px.pie(st.session_state.df, names=x, values=y)
                    elif chart_type == "histogram":
                        fig = px.histogram(st.session_state.df, x=x, color=color)
                    else:
                        st.warning("Unknown or unsupported chart type from Gemini.")

                if fig:
                    st.session_state.chat_history.append({"role": "plot", "content": fig})
            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ Error creating visualization: {e}"})

        else:
            # Normal text Q&A: ask Gemini to analyze the sample and answer the question
            prompt = f"""
You are a data analyst assistant. Use the dataset sample below and answer the user's question concisely.

Dataset sample:
{sample_data}

User question: {user_input}

Answer in plain text.
"""
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(prompt).text.strip()
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ Gemini error: {e}"})

    # Display chat history and any generated plots
    if "chat_history" in st.session_state:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            elif message["role"] == "assistant":
                st.chat_message("assistant").write(message["content"])
            elif message["role"] == "plot":
                st.plotly_chart(message["content"], use_container_width=True)


if __name__ == "__main__":
    # Use the password check pattern from your example file
    if check_password():
        run_data_bot()
