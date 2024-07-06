#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install streamlit


# In[15]:


import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import os
import openai
import sys
import datetime

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import OpenAIEmbeddings
from YT_downloads import download_videos,process_videos,convert_to_text   #importing functions from YT_dowmload file


# In[16]:


directory = r'your_working_directory'

# Change the current working directory to the specified directory
os.chdir(directory)
download_path = 'your_working_directory/download_folder' #Folder to store downloaded YT videos
persist_directory = 'your_working_directory/download_folder/cromadb_dir2' #VectorDB folder

# In[17]:


sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
os.environ["OPENAI_API_KEY"] = "your_openai_key"





# In[29]:


# Function to load the vectorDB
def load_vectordb(persist_directory,embedding):
    
    return Chroma(persist_directory=persist_directory, embedding_function=embedding)


# In[30]:


# Function to retrieve information using vectorDB
def retrieve_info(question, vectordb, my_llm, my_memory):
    qa_chain =  ConversationalRetrievalChain.from_llm(
        llm=my_llm, #Your choice of LLM
        memory=my_memory, #memeory is used from langchain. There are a few different types of memories you can use. 
        chain_type="stuff", #Stuff is the basic one- meaning providing all context information at once to llm. There are other types - map_reduce (breaks info into stages and then combine results from all stages), refine (start with an initial anaswer and then refine in subsequent turns) and map_rerank(selecting the best answer)
        retriever=vectordb.as_retriever(),
        return_source_documents=True, #Ensures that the source documents used to generate the response are returned, which can be useful for validation and reference.
        get_chat_history=lambda h : h, # A lambda function to retrieve the chat history
        verbose=False)

    response = qa_chain.invoke({"question": question})
    return response

# In[37]:

#using streamlit to desing UI.
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
            return_messages=True  # Indicates that the full conversation messages should be returned, not just the summaries
        )
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    st.write("### Chat History")
    for i, message in enumerate(st.session_state.chat_history):
        st.write(f"{i + 1}. {message['user']}: {message['text']}")
        #Uncomment next line if you want to display source document too.
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
                combined_file_name=convert_to_text()
                process_videos(combined_file_name)
                st.success("Videos downloaded and processed successfully!")
                
        else:
            st.error("Please enter at least one video URL.")

if __name__ == "__main__":
    main()


# In[ ]:




