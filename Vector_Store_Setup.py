from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Load Environment Variables
load_dotenv()

# Get API Keys from Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Validate API Keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found. Please check your .env file.")

# Set Environment Variables
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

def create_pinecone_index():
    """Create Pinecone Index one-time"""
    print("Creating Pinecone index...")
    
    # Load Data
    loader = TextLoader('Data.txt', encoding='utf-8')
    docs = loader.load()
    
    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(docs)
    
    # Create Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create Vector Store
    index_name = "pinecone-chatbot"
    vectorstore = PineconeVectorStore.from_documents(
        split_docs, 
        embeddings, 
        index_name=index_name
    )
    
    print(f"Index '{index_name}' successfully created!")
    return vectorstore

if __name__ == "__main__":
    create_pinecone_index()