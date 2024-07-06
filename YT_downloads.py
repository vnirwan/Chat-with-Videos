#!/usr/bin/env python
# coding: utf-8

# This notebook is to download YT mp4 videos

# In[ ]:


#!pip install python-dotenv


# In[1]:

import os
import youtube_dl
import whisper
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import shutil

# Set the directory 
directory = r'your_working_directory'

# Change the current working directory to the specified directory
os.chdir(directory)


# In[4]:

# use pytube to download videos from YT
from pytube import YouTube
# Specify the directory where you want to save the videos
download_path = 'your_working_directory/download_folder'
def download_videos(video_urls):
    """
    This function downloads YT videos to your specified folder
    """
    subfolder_path ='your_working_directory/download_folder/old_videos' # to move the existing videos to an old folder, so that we only process the new videos
    # List all files in the main folder
    files = [f for f in os.listdir(download_path) if os.path.isfile(os.path.join(download_path, f))]

# Move each file to the subfolder, so that we only run our code for the new YT url 
    for file in files:
        src_path = os.path.join(download_path, file)
        dst_path = os.path.join(subfolder_path, file)
        shutil.move(src_path, dst_path)
    
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    for video_url in video_urls:
        try:
            # Create a YouTube object
            yt = YouTube(video_url)

            # Choose the stream you want to download (e.g., the highest resolution stream)
            stream = yt.streams.get_highest_resolution()

            # Download the video to the specified directory
            stream.download(output_path=download_path)

            print(f'Video from URL {video_url} downloaded successfully!')
        except Exception as e:
            print(f'Error downloading video from URL {video_url}: {str(e)}')
            
def convert_to_text():
    """
    This function converts videos to audio,
    pre-process it,
    converts into text and
    saves in a txt file to be used by process_videos function
    """
    folder_path = download_path #can also pass it as a parameter
    #using whisper to convert audio to text
    model = whisper.load_model("base")
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        file_names = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mp4')]
    else:
        print(f"The folder '{folder_path}' does not exist or is not a directory.")
        return
    
    combined_text = ""
    
    for file_name in file_names:
        audio = whisper.load_audio(file_name)
        audio = whisper.pad_or_trim(audio) #trimming the audio file, to fasten the processing for this demo. You can remove this step or add specific length.
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
        text = result.text
        combined_text += f'Video_title: {file_name} {text}\n' #adding video title to the result, so that we can validate which video is the source of answers. 
    
    #Name of the text file. You can also pass it as a parameter in this function.
    combined_file_name = 'combined_video_data_demo.txt' 
    #Check if combined_video_data_demo is already there which will have old data in it. If yes, we delete it and create a new one to avoid any duplicates
    #Can be extended to check if a video has already been downloaded or processed already in this DB
    if os.path.exists(combined_file_name):
        os.remove(combined_file_name)
    with open(combined_file_name, 'w', encoding='utf-8') as file:
        file.write(combined_text + '\n')
    return (combined_file_name)

def process_videos(combined_file_name):
    """This function reads a text file, 
    splits the content of this file into managable chucks, 
    converts into embeddings and
    stores in a vectorDB
    """
   
    loader = TextLoader(combined_file_name, encoding='utf-8') # load the text file from convert_to_text function
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



if __name__ == "__main__":
    #Example YT urls
    video_urls = [
    
]
    download_videos(video_urls)
    convert_to_text()
    process_videos(combined_file_name)





# In[ ]:




