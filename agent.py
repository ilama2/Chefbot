# agent.py
import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import Tool, initialize_agent, AgentType
from vector_store import query_vector_store, rerank_results, format_context

load_dotenv()

# Initialize OpenAI and Pinecone clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")

# Define the index name used in Pinecone for storing transcripts
index_name = "audios-transcripts"

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Set a maximum token limit for context to avoid exceeding model limits
MAX_TOKENS = 1500  

# Function to generate a structured prompt 
def generate_prompt(context, recipe_name, question):
    # Constructing the prompt to be sent to the agent
        prompt = f"""
        You are a professional virtual chef assistant, dedicated to offering thorough, systematic, and precise cooking guidance. Your goal is to help the user by answering questions based on a YouTube cooking video transcript and providing detailed responses for general cooking queries with step-by-step explanations.

        - **For video-related questions**: Answer the user's question about {recipe_name} **using only the information from the provided transcript**. If the transcript doesn’t provide enough details, do the following:
        1. Acknowledge the limitation of the transcript.
        2. Provide **any relevant information available** from the transcript that can still be useful.
        3. Offer **general advice based on culinary knowledge** if necessary.

        - **For general cooking questions**: If the question is not related to the transcript or video content, provide a **structured response** that includes:
        1. **Ingredient breakdown**: Clearly list and explain each ingredient, including its role in the recipe.
        2. **Step-by-step instructions**: Outline each step of the process, providing detailed reasoning behind each action. For example, why certain ingredients are added at specific times, or how to adjust cooking times.
        3. **Techniques and tips**: Explain any specific techniques (e.g., creaming butter, folding ingredients) and offer tips for achieving the best results. Address common mistakes and how to avoid them.
        4. **Cooking times and temperatures**: Be clear about **recommended cooking times and temperatures**, and suggest adjustments based on factors like altitude, oven type, or ingredient variations.
        5. **Tools and equipment**: Highlight any important tools or equipment used in the recipe, and provide tips for their optimal use (e.g., type of pan, temperature of ingredients).
        6. **Substitutions or variations**: Provide suggestions for ingredient substitutions if the user doesn’t have certain items.
        7. **Serving suggestions**: Offer ideas for serving, plating, and pairing with other dishes.

        Ensure that the response is **comprehensive**, **systematic**, and **easy to follow**. Use clear formatting like bullet points or numbered lists to make the information digestible.

        Maintain a **friendly and approachable** tone, and use **creative emojis** where appropriate to enhance the response.

        Context from video transcript:
        {context}

        User's Question: {question}

        Answer:
        """
        return prompt

# Retrieve relevant context and recipe name from the video transcript
def get_video_context(question, video_id, client, pc, index_name):
    # Step 1: Try retrieving context from the video
    results = query_vector_store(question, client, pc, index_name, video_id, top_k=8)
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


        return context, recipe_name

# Utility function to limit context size based on word count
def truncate_context(context, max_tokens=MAX_TOKENS):
    # Truncate the context to the desired length
    tokens = context.split()  # Tokenize by whitespace
    truncated_context = " ".join(tokens[:max_tokens])  # Keep the first `max_tokens` tokens
    return truncated_context

# Tool function used by the agent to fetch context and generate an answer
def recipe_tool(query, video_id, client, pc, index_name):
    """Tool function to retrieve context and generate prompt for recipe question."""
    context, recipe_name = get_video_context(query, video_id, client, pc, index_name)

    if context and recipe_name:
        context = truncate_context(context)  # Truncate context if needed
        prompt = generate_prompt(context, recipe_name, query)

        # Create the message for the language model
        messages = [
            SystemMessage(content="You are a warm and expert virtual chef assistant."),
            HumanMessage(content=prompt)
        ]
        
        # Get the response from the language model
        response = llm.invoke(messages)
        
        # Return both the context (Observation) and the final answer
        return context, response.content
    else:
        return "Sorry, I couldn't find relevant information in the video transcript.", None

# Main function that sets up and runs the LangChain agent to handle user questions
def get_recipe_answer(question: str, video_id: str, client, pc, index_name: str) -> str:
    # Define the tool the agent can use for recipe Q&A
    tools = [
        Tool(
            name="RecipeQuestion",
            func=lambda query: recipe_tool(query, video_id, client, pc, index_name),  # Pass `video_id` here for each call
            description="Use this tool to answer any questions about the cooking video or recipes."
        )
    ]

    # Initialize the LangChain agent with the defined tool
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True
    )

    # Run the agent with the user's question 
    return  agent.run(question)

# The agent doesn't work as expected. 
# It retuns short answer and not the full answer.
