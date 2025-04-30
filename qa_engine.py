from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from vector_store import query_vector_store, rerank_results, format_context, get_embedding

# Initializes the custom retriever
class PineconeHybridRetriever:
    def __init__(self, client, pc, index_name, video_id, recipe_name=None):
        self.client = client
        self.pc = pc
        self.index_name = index_name
        self.video_id = video_id
        self.recipe_name = recipe_name

    # Retrieves relevant documents from Pinecone based on the query
    def get_relevant_documents(self, query, top_k=5):

        # Get the embedding for the query
        query_embedding = get_embedding(query, self.client)
        
        # Define the Pinecone index
        index = self.pc.Index(self.index_name)
        
        # Prepare the query params for Pinecone
        query_params = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True
        }
        
        # Add the filter for recipe_name if it's provided
        if self.recipe_name:
            query_params["filter"] = {"recipe_name": {"$eq": self.recipe_name}}
        
        # Execute the query
        results = index.query(**query_params)
        
        # Return the matches found in Pinecone
        return [Document(page_content=match["metadata"]["text"], metadata=match["metadata"]) for match in results["matches"]]
    

def ask_question_with_video_context(question, video_id, client, pc, index_name, recipe_name):
    # Initialize the PineconeHybridRetriever to fetch relevant documents
    retriever = PineconeHybridRetriever(client, pc, index_name, video_id, recipe_name)
    
    # Use the retrieved documents as context for the LLM
    docs = retriever.get_relevant_documents(question)
    context = format_context(docs)  # Format the context
    
    # Create a prompt template for answering the question
    prompt = PromptTemplate(
        input_variables=["context", "question", "recipe_name"],
        template="""  
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

        Answer:"""
    )
    
    # Initialize the LLM (use ChatOpenAI with your desired model and parameters)
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # Create an LLMChain with the prompt and LLM
    chain = LLMChain(prompt=prompt, llm=llm)
    
    # Generate the answer based on the context and question and recipe name
    answer = chain.run({"context": context, "question": question, "recipe_name": recipe_name})
    
    return answer

def ask_question_general(question, client, pc, index_name):
    retriever = PineconeHybridRetriever(client, pc, index_name)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a knowledgeable assistant. Please answer this question using only the provided context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # Initialize LLM and chain
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    chain = LLMChain(prompt=prompt, llm=llm)
    
    # Get relevant documents from Pinecone
    docs = retriever.get_relevant_documents(question)
    context = format_context(docs)  # Adjust the context as necessary

    # Get the answer from the LLM
    answer = chain.run({"context": context, "question": question})
    return answer

