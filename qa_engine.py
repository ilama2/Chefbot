from vector_store import query_vector_store, rerank_results, format_context

def ask_question_with_video_context(question, video_id, client, pc, index_name):
    """Answer question using RAG with video transcript context."""
    # Get relevant chunks
    results = query_vector_store(question, client, pc, index_name, video_id, top_k=8)
    
    if not results:
        #return "I couldn't find any relevant information to answer your question. Could you provide more details or ask a different question?"
        return ask_question_general(question, client)
    
    # Rerank results
    reranked_results = rerank_results(question, results)
    
    # Format context
    context = format_context(reranked_results[:5])  # Use top 5 after reranking
    
    # Try to identify the recipe name for better context
    recipe_names = [r["metadata"].get("recipe_name", "").lower() for r in reranked_results 
                   if "recipe_name" in r["metadata"]]
    recipe_name = max(set(recipe_names), key=recipe_names.count) if recipe_names else "this recipe"
    
    # Get video ID from results if not provided
    if not video_id and results:
        video_id = results[0]["metadata"].get("video_id", "unknown")
    
    # Create prompt for LLM
    prompt = f"""
You are a warm and expert virtual chef assistant. Your job is to help the user based on a YouTube cooking video transcript.

Answer the user's question about {recipe_name} using ONLY the information from the provided context.

If the context doesn't contain enough information to answer fully, acknowledge this limitation and provide what you can from the context.
If you cannot answer the question from the context, say so clearly rather than making up information.

Be especially careful with:
- Quantities, measurements, and proportions
- Cooking times and temperatures 
- Techniques and special instructions
- Equipment and tools mentioned

Provide your answer in a friendly, confident chef's voice.

Context from video transcript:
{context}

User's Question: {question}

Answer:
"""
    
    # Get answer from LLM
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a friendly and knowledgeable recipe assistant who helps users understand cooking steps based on YouTube video transcripts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content

def ask_question_general(question, client):
    """Answer general cooking questions without specific video context."""
    prompt = f"""
You are a helpful cooking assistant with broad knowledge about cooking techniques, ingredients, and recipes.

Question: {question}

Please provide a helpful, accurate response. If the question requires specific recipe details you don't have, provide general guidance instead.

Answer:
"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a friendly recipe expert with years of cooking experience."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=400
    )
    
    return response.choices[0].message.content

