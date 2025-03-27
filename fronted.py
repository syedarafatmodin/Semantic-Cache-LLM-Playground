import streamlit as st
import requests
from datetime import datetime

# Configure page
st.set_page_config(page_title="Semantic Cache QA", layout="wide")
st.title("Semantic Cache QA System")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    backend_url = st.text_input(
        "Backend URL",
        value="http://localhost:8000",
        help="URL of the backend server"
    )
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.7,
        max_value=0.95,
        value=0.85,
        step=0.01,
        help="Minimum similarity score to use cached answer"
    )

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "details" in message:
            with st.expander("Details"):
                st.json(message["details"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{backend_url}/ask",
                    json={"question": prompt}
                ).json()

                st.markdown(response["answer"])
                
                details = {
                    "source": response["source"],
                    "timestamp": response["timestamp"]
                }
                if response["source"] == "cache":
                    details.update({
                        "similarity": response["similarity"],
                        "matched_question": response["matched_question"]
                    })

                with st.expander("Response Details"):
                    st.json(details)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "details": details
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {str(e)}"
                })