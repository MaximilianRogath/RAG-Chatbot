from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import gradio as gr

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

def load_existing_conversation_system():
    """Load existing Pinecone Index"""
    print("Loading existing Pinecone index...")
    
    try:
        # Embeddings (same as used in setup)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Load existing vector store
        index_name = "pinecone-chatbot"
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )
        
        # Initialize Memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create conversation system
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            verbose=True
        )
        
        print("Conversation system loaded successfully!")
        return conversation_chain
        
    except Exception as e:
        print(f"Error loading index: {e}")
        print("Please run 'python setup_index.py' first to create the index.")
        raise

# Load conversation system (not create!)
conversation_system = load_existing_conversation_system()

# Function for Gradio Interface
def respond(message, history):
    # Send request to conversation system
    response = conversation_system.invoke({"question": message})
    
    # Extract answer
    if isinstance(response, dict) and 'answer' in response:
        return response['answer']
    else:
        return str(response)

# Create Gradio Interface
demo = gr.ChatInterface(
    fn=respond,
    title="Virtual Assistant Aschaffenburg UAS",
    description="Welcome to the virtual assistant of Aschaffenburg University of Applied Sciences. Please refrain from sharing any personal or sensitive information. This assistant is designed to answer general questions about our university and services. How can I help you today?",
    theme=gr.themes.Soft(),
    examples=[
        "What is the number of students at the university?",
        "Which degree programmes are offered?",
        "When was Aschaffenburg UAS founded?",
        "What are the library's opening hours?"
    ],
    retry_btn=None,
    undo_btn=None
)

# Start app
if __name__ == "__main__":
    demo.launch(share=True)