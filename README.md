# Chefbot Multimodal Cooking Assistant

## Setup Instructions:
1. Prerequisites
   - python 3.9+
   -  FFmpeg installed on the system
   -  OpenAI API key
   -  Pinecone API key
   -  LangSmith API key (for tracing)
2. Environment Setup
   -  Clone the repository:
     ```bash
     git clone https://github.com/ilama2/Chefbot.git
     cd Chefbot
3

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
   - 
     
     
     
     
  
