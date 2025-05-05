# Download audio -> Convert to mp3 - > Transcribe audio
import os
import subprocess
import re
from typing import Optional
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
import whisper
from urllib.parse import urlparse, parse_qs


# Extract YouTube video ID from various URL formats.
def extract_video_id(url):
    parsed_url = urlparse(url)

    # Case 1: Standard YouTube URL
    if "youtube.com" in parsed_url.netloc:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    # Case 2: Shortened YouTube URL
    elif "youtu.be" in parsed_url.netloc:
        return parsed_url.path.lstrip("/")

    return None

# Download audio from YouTube using yt-dlp.
def audio_downloader_yt_dlp(youtube_url):
    try:
        # Define download options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'audio_%(id)s.%(ext)s',
            'quiet': False,
            'extractaudio': True,
            'audioquality': 0,
            'postprocessors': [],
        }

        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            video_id = info_dict['id']
            # Find the downloaded file
            for file in os.listdir():
                if file.startswith(f"audio_{video_id}"):
                    return file
            return None
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

# Convert .webm to .mp3 using ffmpeg.
def convert_webm_to_mp3(webm_file):
    mp3_file = webm_file.replace('.webm', '.mp3')
    try:
        # Run ffmpeg to convert .webm to .mp3
        subprocess.run(['ffmpeg', '-i', webm_file, mp3_file], check=True)
        return mp3_file
    except Exception as e:
        print(f"Error converting file: {e}")
        return None

# Clean transcript text by removing filler words and normalizing spacing.
def clean_transcript(raw_text):
    text = raw_text.replace("\\'", "'").replace('\\"', '"')
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s([.,!?;:])', r'\1', text)

    filler_words = [
        'okay', 'hmm', 'ah', 'um', 'uh', 'mm', '(laughs)',
        '(chimes ringing)', '(knife tapping cutting board)',
        '(fire truck siren blaring)', '(smoke alarm beeping)', '(Music)'
    ]
    for word in filler_words:
        text = text.replace(word, '')
    
    return text.strip()

# Transcribe audio using Whisper.
def transcript_whisper(file_audio):
    try:
        model = whisper.load_model("large")
        result = model.transcribe(file_audio)
        return clean_transcript(result['text'])
    except Exception as e:
        print(f"Error transcribing with Whisper: {e}")
        return None
    
# Main function to get video transcript.
def get_video_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        print("Error: Invalid YouTube URL")
        return None

    os.makedirs('transcripts', exist_ok=True)

    # Download audio using yt-dlp
    audio_file = audio_downloader_yt_dlp(video_url)
    if not audio_file:
        print("Error: Failed to download audio.")
        return None

    # Convert to mp3
    mp3_file = convert_webm_to_mp3(audio_file)
    if not mp3_file:
        print("Error: Failed to convert audio.")
        return None

    # Transcribe using Whisper
    transcript = transcript_whisper(mp3_file)
    if not transcript:
        print("Error: Failed to transcribe audio.")
        return None

    # Cleanup downloaded audio files
    for f in [audio_file, mp3_file]:
        if f and os.path.exists(f):
            os.remove(f)

    # Save transcript to disk
    transcript_path = f"transcripts/transcript+{video_id}.txt"
    with open(transcript_path, "w") as f:
        f.write(transcript)

    print(f"Transcript saved to {transcript_path}")
    return transcript

