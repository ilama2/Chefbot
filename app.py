# 📂 Same imports as before (no change needed)
import streamlit as st
import time
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import requests
from urllib.parse import urlparse
from pinecone import Pinecone
from langchain_openai import ChatOpenAI

from rag import ask_question
from transcript_processor import get_video_transcript, extract_video_id
from vector_store import add_video_to_vectorstore, create_pinecone_index_if_needed
import os
import subprocess
import ffmpeg


os.system('apt-get update')
os.system('apt-get install -y ffmpeg')
import asyncio

# Ensure an event loop is running
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# 🔁 Load environment
load_dotenv()
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=pinecone_api_key, environment="us-east-1")
index_name = "audios-transcripts"
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="ChefBot", page_icon="👨‍🍳", layout="wide")

# ✅ Session state
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# ✅ Initialize Pinecone
with st.spinner("Initializing system..."):
    create_pinecone_index_if_needed(pc, index_name)

# ✅ Page header
st.markdown("<h1 style='color:#FF4B4B;'>ChefBot 👨‍🍳🤖</h1>", unsafe_allow_html=True)
st.markdown("Analyze cooking videos or ask general culinary questions!", unsafe_allow_html=True)

# ✅ Sidebar
with st.sidebar:
    st.header("About ChefBot 👨‍🍳")
    st.write("- Smart kitchen assistant using AI & video transcripts.")
    st.write("- Ask about ingredients, steps, or any cooking topic.")
    st.divider()
    st.header("📝 How to Use")
    st.write("1. Paste a **YouTube cooking video link**.")
    st.write("2. Click **Process Video**.")
    st.write("3. Ask **anything cooking-related**!")
    st.divider()
    st.write("No video? No problem — just ask general cooking questions!")

# ✅ YouTube helpers
def extract_thumbnail(video_id):
    urls = [f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
            f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"]
    for url in urls:
        r = requests.get(url)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content))
    return None

def embed_video(video_id):
    return f"""
    <iframe width="100%" height="315" src="https://www.youtube.com/embed/{video_id}"
     frameborder="0" allowfullscreen style="border-radius: 10px;"></iframe>
    """

# ✅ Main Input: Process YouTube
st.markdown("---")
video_url = st.text_input("🔗 Enter YouTube URL (optional)", placeholder="https://www.youtube.com/watch?v=...")
if st.button("🎥 Process Video"):
    vid = extract_video_id(video_url)
    if not vid:
        st.error("Invalid YouTube URL.")
    else:
        st.session_state.current_video_id = vid
        with st.spinner("Fetching transcript..."):
            transcript = get_video_transcript(video_url)
            if transcript:
                st.session_state.transcript = transcript
                with st.spinner("Indexing transcript..."):
                    add_video_to_vectorstore(transcript, vid, pc, index_name)
                st.session_state.video_processed = True
                st.success("Video processed and ready!")
            else:
                st.error("Transcript unavailable for this video.")

# ✅ Display Video / Thumbnail
if st.session_state.current_video_id:
    st.image(extract_thumbnail(st.session_state.current_video_id), use_container_width=True)
    st.markdown(embed_video(st.session_state.current_video_id), unsafe_allow_html=True)

# ✅ Ask a cooking question
st.markdown("---")
st.markdown("## Ask ChefBot a Question 🍽️")
user_q = st.text_input("🧑‍🍳 Your Question", placeholder="e.g. How long should I bake lasagna?")
if st.button("❓ Ask"):
    with st.spinner("Thinking..."):
        video_id = st.session_state.current_video_id if st.session_state.video_processed else None
        reply = ask_question(user_q, video_id, llm, pc, index_name)
        st.session_state.conversation_history.append({"question": user_q, "answer": reply})

# ✅ Conversation History
st.markdown("---")
st.markdown("## 📜 Conversation History")
if st.session_state.conversation_history:
    for i, item in enumerate(st.session_state.conversation_history):
        st.markdown(f"**Question{i+1}: {item['question']}**")
        st.markdown(f"<div style='background:#f9f9f9;padding:10px;border-left:4px solid #FF4B4B;'>{item['answer']}</div>", unsafe_allow_html=True)
else:
    st.info("No questions asked yet!")

# ✅ Transcript Viewer
with st.expander("📄 View Transcript"):
    if st.session_state.transcript:
        st.text_area("Transcript", st.session_state.transcript, height=300)
    else:
        st.info("No transcript loaded.")
