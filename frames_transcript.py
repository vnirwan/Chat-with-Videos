import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import os
import openai
import sys
import datetime
import time
import cv2
import base64
from IPython.display import display, Image
import whisper

import youtube_dl
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import shutil
import yt_dlp


directory = r'your_directory'

# Change the current working directory to the specified directory
os.chdir(directory)

#Function to download videos
def download_videos(video_urls):
    # Create download path and subfolder if they don't exist
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # List all files in the main folder
    files = [f for f in os.listdir(download_path) if os.path.isfile(os.path.join(download_path, f))]

    # Move each file to the subfolder, so that we only run our code for the new YT URL
    for file in files:
        src_path = os.path.join(download_path, file)
        dst_path = os.path.join(subfolder_path, file)
        shutil.move(src_path, dst_path)
    
    # Define download options
    ydl_opts = {
        'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(download_path, '%(title)s.%(ext)s'),  # Save with the title of the video as filename
    }
    
    # Function to download a video from a given URL
    def download_video_yt_dlp(url):
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    # Download each video in the list
    for video_url in video_urls:
        try:
            print(f"Downloading video from URL: {video_url}")
            download_video_yt_dlp(video_url)
            print(f"Successfully downloaded video from URL: {video_url}")
        except Exception as e:
            print(f'Error downloading video from URL {video_url}: {str(e)}')


#Convert video to frames
#video_path exmaple -> video_path = "INSANE 90+ TOTS Pack! 😱 #shorts.mp4"
def convert_video_to_frames(video_path):
    """
    This function converts video into multiple frames.
    """
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    print(len(base64Frames), "frames read.")
    return base64Frames



#to get summary of frames
def generate_summary_from_frames(base64Frames):
    """
    This function gets summary from different frames.Uses gpt 4o model to get summary.
    """
    #Uncomment following code to see different frames produced.
    #display_handle = display(None, display_id=True)
    #for img in base64Frames:
        #display_handle.update(Image(data=base64.b64decode(img.encode("utf-8"))))
        #time.sleep(0.025)

    
    #You can customize this prompt according to your needs.
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "These are frames from a video that I want to get a summary for. It should capture all the things happening in the video. Add a sentiment score to the video.",
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50]),
            ],
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 200,
    }
    result = openai.chat.completions.create(**params)
    summary = result.choices[0].message.content
    return summary

# Generate video transcript
def generate_video_transcript(file_name):
    """
    This function converts videos to audio,
    pre-process it,
    converts into text.
    """
    #using whisper to convert audio to text
    model = whisper.load_model("base")
    combined_text = ""
    audio = whisper.load_audio(file_name)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    text = result.text
    combined_text += f'Video_title: {file_name} {text}\n' 
    return combined_text


# Combine summary and transcript into a single file
def combine_summary_and_transcript(summary, transcript, output_file):
    print("In combine_summary_and_transcript function")
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("Video Summary:\n")
        file.write(summary + '\n\n')
        file.write("Video Transcript:\n")
        file.write(transcript + '\n')
        print("combine_summary_and_transcript function done")


def process_videos(combined_file_name):
    """This function reads a text file, 
    splits the content of this file into managable chucks, 
    converts into embeddings and
    stores in a vectorDB
    """
    #define a directory for your vector DB
    persist_directory = 'your_directory/cromadb_dir2'
    #combined_file_name = 'combined_video_data_demo.txt'
    loader = TextLoader(combined_file_name, encoding='utf-8')
    pages = loader.load()
    
    #This step splits the text file into chunks. Chunk_overlap will make sure we don't loose any semantic meaning or continuition of a sentence.
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len
    )
    splits = text_splitter.split_documents(pages)
    #using openAI's embeddings function to convert text to embeddings
    embedding = OpenAIEmbeddings()


    # Check if the directory exists to decide whether to load or create the vector database
    if os.path.exists(persist_directory):
        print('Inside existing cromadb')
        vectordb = Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory
        )
        vectordb.add_documents(splits)
    else:
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=persist_directory
        )

    # Persist the updated database
    vectordb.persist()

