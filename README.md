# Chat-with-Videos
This project uses RAG, LangChain, ChatGPT (clip model), and a vector database to enable users to chat with videos instead of watching them. Users can query video content through conversational interaction, retrieving specific and **correct** information efficiently without viewing the entire video.
## Key Features
Conversational Interaction: Users can query video content through a chat interface, making it easy to find precise information without watching the full video.
Retrieval-Augmented Generation (RAG): Combines retrieval and generation capabilities to provide accurate and contextually relevant answers. It preserves the semantic meaning of sentences. 
LangChain Integration: Facilitates seamless conversation management and processing.
ChatGPT: Powers the natural language understanding and response generation, ensuring a smooth and intuitive user experience.
CLIP (Contrastive Language–Image Pretraining) Model: developed by OpenAI, connect images and text using a contrastive learning approach, enabling tasks like zero-shot learning, image classification, and multi-modal applications with high flexibility and generalization.
Vector Database: Efficiently stores and retrieves video content based on user queries.
Memory Functionality: The user interface is designed using Streamlit, chosen for its simplicity and rapid prototyping capabilities, enabling an interactive and user-friendly experience.

## Step-by-Step Process
### Run the Application:
Open your terminal and navigate to the project directory.
Run the application using the command: streamlit run app.py.

### Add Videos for Processing:
In the application interface, you will find an input box where you can add list of videos separated by commas.
Enter the video URLs or file paths and click on the "Download Videos" button. (Uses frames_transcript.py file) 
Once Video is downloaded, click on "Process video" button.(Uses frames_transcript.py file) 
The application will process these videos in the background and save all relevant information in a vector database.

### Ask Questions:
Once the videos are processed, you can start asking questions about the video content.
Use the first input box to type your question and click on the "Submit" button.
The application will retrieve and display the relevant information based on your query.

### Follow-up Questions:
If you have follow-up questions, simply edit the text in the input box to reflect your new question.
Click on the "Submit" button again to receive updated responses.
The application will display the entire chat history, maintaining memory of the conversation context.
