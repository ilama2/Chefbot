import streamlit as st
import time
import os
import re
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import requests
from urllib.parse import urlparse, parse_qs
import base64
from pinecone import Pinecone
from openai import OpenAI
import pinecone
from rag_system import  ask_question
from transcript_processor import get_video_transcript, extract_video_id
from vector_store import add_video_to_vectorstore, create_pinecone_index_if_needed
from agent import get_recipe_answer

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")
index_name = "audios-transcripts"
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
index = pc.Index(index_name)



# Configure Streamlit page
st.set_page_config(
    page_title="ChefBot",
    page_icon="👨‍🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
    }
    .subheader {
        font-size: 1.5rem;
        color: #666;
    }
    .video-container {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .recipe-container {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #FF4B4B;
    }
    .step-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 1rem;
        color: #FF4B4B;
    }

    .stButton>button:hover {
    background-color: #D94F4F;
    
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states : keeps the app’s memory consistent during user interaction:
if 'current_video_id' not in st.session_state:
    st.session_state.current_video_id = None
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
# Stores the fetched transcript text
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
# Placeholder for extra metadata
if 'video_info' not in st.session_state:
    st.session_state.video_info = {}
# Saves all previous questions and answers in the session.
if 'conversation_history' not in st.session_state: #
    st.session_state.conversation_history = []


# Initialize Pinecone index
with st.spinner("Initializing system..."):
    create_pinecone_index_if_needed(pc, index_name)

# App header
st.markdown("<h1 class='main-header'> ChefBot 👨‍🍳🤖</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Analyze cooking videos and ask questions about recipes</p>", unsafe_allow_html=True)


# Sidebar
with st.sidebar:    
    st.header("About ChefBot 👨‍🍳🤖")
    st.write("""
    - ChefBot is your smart kitchen assistant! 
    - Just paste a YouTube cooking video link, and ChefBot will analyze it for you.  
    - You can then ask detailed questions about the recipe, ingredients, or steps.
    """)
    st.divider()

    st.header("📝 How to Use")
    st.write("1. 🔗 Paste a **YouTube video URL** ")
    st.write("2.  ▶️ Click **Process Video** ")
    st.write("3. 🤖 Ask **For any recipes**")

    st.divider()

    st.header("General Questions")
    st.write("You can also ask general cooking questions without a video.")

# Function to get YouTube thumbnail
def get_youtube_thumbnail(video_id):
    try:
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        response = requests.get(thumbnail_url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            # Try the default thumbnail if maxresdefault is not available
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
            response = requests.get(thumbnail_url)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        return None
    except Exception as e:
        st.error(f"Could not fetch thumbnail: {e}")
        return None

# Function to embed YouTube video
def embed_youtube_video(video_id):
    return f"""
    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; border-radius: 10px;">
        <iframe 
            src="https://www.youtube.com/embed/{video_id}" 
            style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0;" 
            allowfullscreen>
        </iframe>
    </div>
    """

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    # Video URL input
    video_url = st.text_input("Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    process_button = st.button("Process Video", type="primary")
    
    # Process the video
    if process_button and video_url:
        video_id = extract_video_id(video_url)
        
        if not video_id:
            st.error("Invalid YouTube URL. Please check the URL and try again.")
        else:
            st.session_state.current_video_id = video_id
            
            with st.spinner("Fetching and processing video transcript..."):
                transcript = get_video_transcript(video_url) 
                if transcript:
                    st.session_state.transcript = transcript
                    st.success("Video transcript fetched successfully!")
                    
                    with st.spinner("Adding to vector database for search..."):
                        add_video_to_vectorstore(transcript, video_id , client, pc, index_name)
                        st.session_state.video_processed = True
                        st.success("Video processed and ready for questions!")
                else:
                    st.error("Could not process the video. It might not have captions available.")

with col2:
    # Display video thumbnail or embed
    if st.session_state.current_video_id:
        thumbnail = get_youtube_thumbnail(st.session_state.current_video_id)
        if thumbnail:
            st.image(thumbnail, use_container_width=True)        
        # Add video embed
        st.markdown(
            embed_youtube_video(st.session_state.current_video_id),
            unsafe_allow_html=True
        )

# Question input section
st.markdown("---")
st.markdown("<h2>Ask about the recipe 👨‍🍳</h2>", unsafe_allow_html=True)

# Questions and answers
question = st.text_input("Ask a question about the recipe or any recipe in general", 
                         placeholder="Example: What ingredients do I need?")
ask_button = st.button("Ask Question", type="primary")

if ask_button and question:
    with st.spinner("Thinking..."):
        # Determine if we have a processed video and ID
        video_id = st.session_state.current_video_id if st.session_state.video_processed else None
        # Ask question using combined logic
        answer = ask_question(question, video_id, client, pc, index_name)
        # Store in conversation history
        st.session_state.conversation_history.append({
            "question": question,
            "answer": answer 
        })

# Display conversation history
st.markdown("---")
st.markdown("<h2>Conversation History</h2>", unsafe_allow_html=True)

if st.session_state.conversation_history:
    for i, exchange in enumerate(st.session_state.conversation_history):
        st.markdown(f"<div class='step-header'>Question {i+1}:</div>", unsafe_allow_html=True)
        st.markdown(f"<div>{exchange['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='step-header'>Answer:</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='recipe-container'>{exchange['answer']}</div>", unsafe_allow_html=True)
        st.markdown("---")
else:
    st.info("Ask a question to see the response here.")

# Transcript viewer (collapsed by default)
with st.expander("View Transcript"):
    if st.session_state.transcript:
        st.text_area("Video Transcript", st.session_state.transcript, height=300)
    else:
        st.info("Process a video to see its transcript here.")

