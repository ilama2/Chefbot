# Chefbot Multimodal Cooking Assistant

## Setup Instructions:
### 1. Prerequisites
   - python 3.9+
   -  FFmpeg installed on the system
   -  OpenAI API key
   -  Pinecone API key
   -  LangSmith API key (for tracing)
### 2. Environment Setup
   -  Clone the repository:
       ```bash
      git clone https://github.com/ilama2/Chefbot.git
      cd Chefbot


   -  Create and activate a virtual environment:
      ```bash

      python -m venv venv
     
      # Activate the virtual environment
      # On macOS/Linux:
      source venv/bin/activate
      # On Windows:
      venv\Scripts\activate

   - Install dependencies:
      ```bash
      pip install -r requirements.txt
   - Create a .env file with your API keys:
     
              OPENAI_API_KEY = your_openai_api_key
              PINECONE_API_KEY = your_pinecone_api_key
              LANGCHAIN_API_KEY = your_langsmith_api_key
              LANGCHAIN_PROJECT = your_langsmith_project


  ## 2. Repository Structure:
        
   <img width="491" alt="Screenshot 1446-11-08 at 15 33 27" src="https://github.com/user-attachments/assets/972932c8-81e7-45e0-8a4f-ef92eb410d7b" />

  ## 3. Configuration Details
   1. requirements.txt
   2. Environment Variables (.env)
     
              OPENAI_API_KEY = your_openai_api_key
              PINECONE_API_KEY = your_pinecone_api_key
              LANGCHAIN_API_KEY = your_langsmith_api_key
              LANGCHAIN_PROJECT = your_langsmith_project
  ## 4. Usage Guide:
### 1.  Running the Application:
       ```bash
       cd chefbot
       streamlit run app.py
###2. Using ChefBot
      1. Process a YouTube Cooking Video:
       - Paste a YouTube URL into input field
       - Click "Process Video" to extract and process the transcript
       - Wait for confirmation that the video has been processed
      2. Ask Questions:
       - Type your question about the recipe
       - Click "Ask Question" to get a response
       - View the answer in the conversation history section
      3. General Cooking Questions:
       - You can ask general cooking questions without processing a video
      4.  Example Questions:
            `How long should I cook the chicken?`,
            `What tools do I need for this recipe?`,
            `What ingredients do I need for this recipe?`,
      5. Viewing Transcripts
       - Click on "View Transcript" to see the full transcript of the processed video

     
     
     
  
