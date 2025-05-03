from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.schema import SystemMessage, HumanMessage
from vector_store import query_vector_store, rerank_results, format_context, get_embedding

#@traceable()
# retrieval-augmented generation (RAG) system
def ask_question(question, video_id, client, pc, index_name):
 
    # Step 1: Try retrieving context from video
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



        # Construct context-aware prompt
        prompt = f"""
        You are a professional virtual chef assistant, dedicated to offering thorough, systematic, and precise cooking guidance. Your goal is to help the user by answering questions based on a YouTube cooking video transcript and providing detailed responses for general cooking queries with step-by-step explanations.

        - **For video-related questions**: Answer the user's question about {recipe_name} **using only the information from the provided transcript**. If the transcript doesn’t provide enough details, do the following:
        1. Acknowledge the limitation of the transcript.
        2. Provide **any relevant information available** from the transcript that can still be useful.
        3. Offer **general advice based on culinary knowledge** if necessary.

        - **For general cooking questions**: If the question is not related to the transcript or video content, provide a **structured response** that includes:
        1. **Ingredient**: Clearly list and explain each ingredient.
        2. **Step-by-step instructions**: Outline each step of the process, providing detailed reasoning behind each action. For example, why certain ingredients are added at specific times, or how to adjust cooking times.
        3. **Techniques and tips**: Explain any specific techniques (e.g., creaming butter, folding ingredients) and offer tips for achieving the best results. Address common mistakes and how to avoid them.
        4. **Cooking times and temperatures**: Be clear about **recommended cooking times and temperatures**, and suggest adjustments based on factors like altitude, oven type, or ingredient variations.
        5. **Tools and equipment**: Highlight any important tools or equipment used in the recipe, and provide tips for their optimal use (e.g., type of pan, temperature of ingredients).
        6. **Substitutions or variations**: Provide suggestions for ingredient substitutions if the user doesn’t have certain items.

        Ensure that the response is **comprehensive**, **systematic**, and **easy to follow**. Use clear formatting like bullet points or numbered lists to make the information digestible.

        Maintain a **friendly and approachable** tone, and use **creative emojis** where appropriate to enhance the response.

        Context from video transcript:
        {context}

        User's Question: {question}

        Answer:
        """
  # Get answer from LLM
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a friendly and knowledgeable recipe assistant who helps users understand cooking steps based on YouTube video transcripts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
        )
    return response.choices[0].message.content


# works really well
