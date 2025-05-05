import os 
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.schema import SystemMessage, HumanMessage
from langsmith import traceable
from pinecone import Pinecone
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import openai
from langsmith.wrappers import wrap_openai
openai_client = wrap_openai(openai.Client())
load_dotenv()
from vector_store import query_vector_store, rerank_results, format_context, get_embedding, add_video_to_vectorstore


LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT =  os.getenv("LANGCHAIN_PROJECT")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")
index_name = "audios-transcripts"
llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"))


# retrieval-augmented generation (RAG) system
@traceable(name="recipe_chatbot_project")
def ask_question(question, video_id,llm, pc, index_name):
 
    # Step 1: Try retrieving context from video
    results = query_vector_store(question, pc, index_name, video_id, top_k=8)
    use_video_context = len(results) > 0

    if use_video_context:
        # Rerank and format video-based context
        reranked_results = rerank_results(question, results)
        context = format_context(reranked_results[:5])
        
        # Try to extract recipe name
        recipe_names = [
            r["metadata"].get("recipe_name", "").strip().lower()
            for r in reranked_results if "recipe_name" in r["metadata"]
        ]
        recipe_names = [name for name in recipe_names if name]
        recipe_name = max(set(recipe_names), key=recipe_names.count) if recipe_names else "this recipe"

        # Create a prompt for the LLM
        prompt = f"""
        You are a friendly and expert virtual chef assistant, here to help with the {recipe_name} based on the YouTube video transcript.
        
        - **For recipe-specific questions**: Provide clear and concise answers directly from the transcript. For time or temperature-related questions, respond briefly, giving only the specific details found in the transcript. If the transcript lacks the info, acknowledge it and provide a general culinary guideline. 

        - **For general cooking advice**: Answer based on your broader cooking knowledge when the question isn’t about the video’s content. Let the user know when the specific details aren’t available and guide them to other resources for more in-depth instructions.

        - **When answering about steps**: Focus on providing **structured, step-by-step** instructions using **numbered lists** or **bullet points**.

        - **Important notes**:
            - Be specific about **cooking times** and **temperatures**.
            - Keep ingredient and instruction answers **brief and clear**.
            - **Avoid combining steps** from other recipes. Stick to the specific recipe in the video.
            - Always provide the most relevant **techniques**, **equipment**, and **ingredients** based on the video.

        Use a friendly chef’s tone and add fun emojis to keep it engaging! 🍳🍴
        If the question is **not** about cooking, food, recipes, or kitchen tips, you not allowed to answer. Kindly say that you are a virtual chef assistant and cannot answer that question.

        Context from the video transcript (if available):
        {context}

        User's Question: {question}

        Answer:
        """
    # Pass in website text
        messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
        ]
  # Get answer from LLM
        response = llm.invoke(messages)
        #return  response.choices[0].message.content
        return response.content
    else:
        # If no context is found, provide a generic response
        return "I couldn't find any relevant information in the video transcript. Please ask a different question or provide more details."
