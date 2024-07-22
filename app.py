
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



directory = r'your_dir'

# Change the current working directory to the specified directory
os.chdir(directory)
persist_directory = 'your_dir/cromadb_dir2'
# Specify the directory where you want to save the videos
download_path = 'your_directory/docs/YT_demo2'
subfolder_path = r'your_directory\YT_demo2\old_videos' #to save already processed videos. You can also delete them


sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
os.environ["OPENAI_API_KEY"] = "your_openai_key"

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import OpenAIEmbeddings
#from YT_downloads import download_videos,process_videos,convert_to_text
import frames_transcript
#Import all functions from frames_transcript file
from frames_transcript import download_videos,convert_video_to_frames,generate_summary_from_frames,generate_video_transcript,combine_summary_and_transcript,process_videos

# Function to load the vectorDB
def load_vectordb(persist_directory,embedding):
    
    return Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Function to retrieve information using the vectorDB
def retrieve_info(question, vectordb, my_llm, my_memory):
    qa_chain =  ConversationalRetrievalChain.from_llm(
        llm=my_llm,
        memory=my_memory,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        get_chat_history=lambda h : h,
        verbose=False)

    response = qa_chain.invoke({"question": question})
    return response


def main():
    st.title("Chat with Videos")

    if not os.path.exists(persist_directory):
        st.error(f"Directory {persist_directory} does not exist. Please run the save_to_vectordb script first.")
        return
    
    # Load vectorDB
    embedding = OpenAIEmbeddings()
    vectordb = load_vectordb(persist_directory, embedding)
    
    current_date = datetime.datetime.now().date()
    if current_date < datetime.date(2023, 9, 2):
        llm_name = "gpt-3.5-turbo-0301"
    else:
        llm_name = "gpt-3.5-turbo"
    my_llm = ChatOpenAI(model_name=llm_name, temperature=0)

    # Initialize memory
    if 'my_memory' not in st.session_state:
        st.session_state.my_memory = ConversationSummaryBufferMemory(
            llm=my_llm,
            output_key='answer',
            memory_key='chat_history',
            return_messages=True
        )
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    st.write("### Chat History")
    for i, message in enumerate(st.session_state.chat_history):
        st.write(f"{i + 1}. {message['user']}: {message['text']}")
        #if 'source_documents' in message:
            #st.write(f"   **Source Documents:** {message['source_documents']}")

    # Handle the user input and responses dynamically
    question = st.text_input("Ask a question:", key="question_input")
    
    if st.button("Submit"):
        if question:
            with st.spinner('Processing...'):
                # Retrieve information
                response = retrieve_info(question, vectordb, my_llm, st.session_state.my_memory)
                
                # Add to chat history
                st.session_state.chat_history.append({"user": "User", "text": question})
                st.session_state.chat_history.append({"user": "Helper", "text": response['answer']})
                
                # Rerun the script to clear the input box
                st.experimental_rerun()

    # Input for video URLs
    video_urls = st.text_input("Enter video URLs separated by commas:")
    
    if st.button("Download Videos"):
        if video_urls:
            with st.spinner('Downloading videos...'):
                urls_list = [url.strip() for url in video_urls.split(',')]
                download_videos(urls_list)
                st.success("Videos downloaded successfully!")
                
        else:
            st.error("Please enter at least one video URL.")
            
    if st.button("Process Videos"):
        with st.spinner('Processing videos...'):
            

            video_files = [f for f in os.listdir(download_path) if f.endswith('.mp4')]
                
            for video_file in video_files:
                video_path = os.path.join(download_path, video_file)
                frames=convert_video_to_frames(video_path)
                summary=generate_summary_from_frames(frames)
                transcript=generate_video_transcript(video_path)
                combined_file_name = 'combined_video_data_demo.txt'
                combine_summary_and_transcript(summary, transcript, combined_file_name)
                process_videos(combined_file_name)
            st.success("Videos processed successfully!")
                
        
if __name__ == "__main__":
    main()
