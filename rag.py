import os 
import json
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import OpenAI
from langchain.schema import SystemMessage, HumanMessage
from langsmith import traceable
from pinecone import Pinecone
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import openai
from langsmith.wrappers import wrap_openai
from langdetect import detect

from vector_store import query_vector_store, rerank_results, format_context, get_embedding, add_video_to_vectorstore

openai_client = wrap_openai(openai.Client())
load_dotenv()

LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")
index_name = "audios-transcripts"
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# --- Feature: Ingredient Extractor for Shopping List ---
def extract_ingredients(text):
    common_ingredients = [
        "salt", "sugar", "oil", "flour", "eggs", "milk", 
        "pepper", "garlic", "onion", "tomato", "butter", "cheese"
    ]
    return [item for item in common_ingredients if item in text.lower()]

# --- Feature: Convert Units ---
def convert_units(text):
    unit_map = {
        "cup": "cup (240ml)",
        "tablespoon": "tablespoon (15ml)",
        "teaspoon": "teaspoon (5ml)"
    }
    for unit, replacement in unit_map.items():
        text = text.replace(unit, replacement)
    return text

# --- Main RAG Function ---
@traceable(name="recipe_chatbot_project")
def ask_question(question, video_id, llm, pc, index_name):
    results = query_vector_store(question, pc, index_name, video_id, top_k=8)
    use_video_context = len(results) > 0

    if use_video_context:
        # Step 1: Process context and recipe name
        reranked_results = rerank_results(question, results)
        context = format_context(reranked_results[:5])
        
        recipe_names = [
            r["metadata"].get("recipe_name", "").strip().lower()
            for r in reranked_results if "recipe_name" in r["metadata"]
        ]
        recipe_names = [name for name in recipe_names if name]
        recipe_name = max(set(recipe_names), key=recipe_names.count) if recipe_names else "this recipe"

        # Step 2: Detect user language
        user_lang = detect(question)
        
        # Step 3: Build system prompt
        prompt = f"""
        You are a friendly and expert virtual chef assistant, here to help with the {recipe_name} based on the YouTube video transcript.

        - For recipe-specific questions: Provide clear and concise answers directly from the transcript. 
        - For time or temperature-related questions: Respond briefly, giving only the specific details found in the transcript.
        - If the transcript lacks the info, acknowledge it and provide a general culinary guideline. 
        - For general cooking advice: Answer using broader cooking knowledge when the question isn’t about the video content. 

        Instructions:
        - Use step-by-step numbered lists when explaining methods.
        - Mention cooking times and temperatures precisely.
        - Do not mix steps from other recipes.
        - Always stick to the transcript and its ingredients/equipment.
        - Keep answers friendly and include fun emojis 🍳🍴

        If the user asks non-cooking-related questions, say you're a chef assistant and cannot help with that.

        Transcript context (if available):
        {context}

        Please answer in {user_lang}.
        
        User's Question: {question}

        Answer:
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]

        # Step 4: Get model response
        response = llm.invoke(messages)

        # Step 5: Auto-generate shopping list
        ingredients = extract_ingredients(response.content)
        if ingredients:
            shopping_list = "\n\n🛒 **Shopping List:**\n" + "\n".join(f"• {item}  " for item in ingredients)
            response.content += shopping_list

        # Step 6: Convert units (optional enhancement)
        response.content = convert_units(response.content)

        return response.content

    else:
        return "I couldn't find any relevant information in the video transcript. Please ask a different question or provide more details."
