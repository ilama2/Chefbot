import time
import re
from openai import OpenAI
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transcript_processor import extract_video_id

# Ensure Pinecone index exists and is ready.
def create_pinecone_index_if_needed(pc, index_name):
    try:
        # Check if index exists by trying to describe it
        try:
            pc.describe_index(index_name)
            print(f"Index '{index_name}' already exists.")
            return
        except Exception:
            # Index doesn't exist, continue to creation
            pass
            
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embeddings dimension
            metric="cosine",
        )
        
        # Wait for the index to be ready
        while True:
            try:
                status = pc.describe_index(index_name).status
                if status["ready"]:
                    print("Index is ready!")
                    break
                else:
                    print("Waiting for index to be ready...")
                    time.sleep(5)
            except Exception as e:
                print(f"Error checking index status: {e}")
                time.sleep(5)
                
    except Exception as e:
        print(f"Error creating index: {e}")

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
# Get embedding
def get_embedding(text):
    return embedding_model.embed_query(text)

# Split text into chunks with overlap
def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return text_splitter.split_text(text)

# Extract metadata from transcript
def extract_metadata(text, video_id):
    # Try to identify the recipe name
    recipe_patterns = [
        r'recipe for ([^\.]+)',
        r'making ([^\.]+) today',
        r'how to make ([^\.]+)',
        r'cooking ([^\.]+) today'
    ]
    
    recipe_name = "unknown recipe"
    for pattern in recipe_patterns:
        match = re.search(pattern, text.lower())
        if match:
            recipe_name = match.group(1).strip()
            break
    
    # Try to extract key ingredients 
    ingredients = []
    ingredient_patterns = [
        r'(\d+[\s\/\-\.]*\d*\s*(?:cup|cups|tablespoon|tablespoons|tbsp|teaspoon|teaspoons|tsp|oz|ounce|ounces|lb|lbs|gram|grams|g|kg|ml|liter|liters|clove|cloves|can|cans)?\s+[^\,\.\n]+)',
        r'(\d+\s+[^\.,]+)'
    ]
    
    for pattern in ingredient_patterns:
        ingredients.extend(re.findall(pattern, text.lower()))
    
    # Limit to avoid noise
    ingredients = ingredients[:15] if ingredients else []
    
    return {
        "recipe_name": recipe_name,
        "ingredients": ingredients,
        "video_id": video_id,
        "source_type": "youtube_transcript"
    }

# Process and add video transcript to Pinecone.
def add_video_to_vectorstore(transcript, video_id,  pc, index_name):
    # Ensure index exists
    #create_pinecone_index_if_needed(pc, index_name)
    # Get index
    index = pc.Index(index_name)
    
    # Extract metadata
    metadata = extract_metadata(transcript, video_id)

    # Word-based chunking
    chunks = split_into_chunks(transcript)
    print(f"Split transcript into {len(chunks)} chunks")

    # Process in batches 
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        vectors_to_upsert = []

        for j, chunk_text in enumerate(batch):
            chunk_index = i + j
            chunk_id = f"{video_id}_{chunk_index}"

            # Get embedding for each chunk
            embedding = get_embedding(chunk_text)

            # Create vector with metadata
            vector = {
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    **metadata,  # Include extracted metadata
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks)
                }
            }
            vectors_to_upsert.append(vector)

        # Upsert batch to Pinecone
        index.upsert(vectors=vectors_to_upsert)
        print(f"Upserted chunks {i} to {i+len(batch)-1}")

        # Sleep to avoid hitting rate limits
        time.sleep(1)
    
    print(f"Successfully added {len(chunks)} chunks to vector store for video {video_id}")

# Query Pinecone for relevant chunks
def query_vector_store(question,  pc, index_name, video_id=None, top_k=5):
    # Get embedding for the question
    query_embedding = get_embedding(question)
    
    # Set up query
    index = pc.Index(index_name)
    query_params = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True
    }
    
    # Add filter if video_id is provided
    if video_id:
        query_params["filter"] = {"video_id": {"$eq": video_id}}
    
    # Execute query
    results = index.query(**query_params)
    
    return results.get("matches", []) 

# Rerank results based on context relevance
def rerank_results(question, results):
    # Simple reranking based on term overlap
    question_terms = set(question.lower().split())
    
    for result in results:
        text = result["metadata"]["text"].lower()
        # Calculate term overlap
        overlap_score = sum(1 for term in question_terms if term in text)
        # Combine with vector similarity for final score
        result["adjusted_score"] = result["score"] * 0.7 + (overlap_score / max(1, len(question_terms))) * 0.3
    
    # Sort by adjusted score
    results.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)
    return results

# Format retrieved results into context for the LLM.
def format_context(results):
    # Simple deduplication by content fingerprinting
    seen_fingerprints = set()
    unique_chunks = []
    
    for result in results:
        text = result["metadata"]["text"].strip()
        # Create a simple fingerprint from the first 100 chars
        fingerprint = text[:100].lower()
        
        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            unique_chunks.append({
                "text": text,
                "score": result.get("adjusted_score", result["score"]),
                "metadata": result["metadata"]
            })
    
    # Format the context
    context_parts = []
    for i, chunk in enumerate(unique_chunks, 1):
        context_parts.append(f"[Chunk {i}]\n{chunk['text']}")
    
    return "\n\n---\n\n".join(context_parts)
