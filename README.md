# RAG-Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with OpenAI and Pinecone. This chatbot can be customized to answer questions about any topic by replacing the data file with your own content.

## Features

* Secure API Key Management - Uses environment variables to protect sensitive credentials
* Efficient Vector Storage - Leverages Pinecone for fast similarity search
* Conversational Memory - Maintains context throughout the conversation
* Modern UI - Clean Gradio interface for easy interaction
* Optimized Performance - Separate indexing and runtime for faster startup

## Architecture

The project is split into two main components:

1. Vector_Store_Setup.py - One-time index creation (run once)
2. main.py - Main chatbot application (run for each session)

## Installation

Clone the repository:
```bash
git clone https://github.com/MaximilianRogath/RAG-Chatbot.git
cd rag-chatbot
```

Install dependencies:
```bash
pip install -r requirements.txt
```
To set up your environment variables, just copy env_example to your .env and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```
 

## Usage

1. First-time setup (creates the vector index):
   ```bash
   python Vector_Store_Setup.py
   ```

2. Run the chatbot:
   ```bash
   python main.py
   ```

3. Access the interface at the URL shown in the terminal.

## Configuration

* Chunk Size: 1000 characters (optimized for general content)
* Chunk Overlap: 200 characters (ensures context preservation)
* Model: GPT-4o for responses, text-embedding-3-small for embeddings
* Vector Store: Pinecone index named "pinecone-chatbot"

## Example Questions

Customize these based on your data content:

* "What information do you have about...?"
* "Can you explain...?"
* "What are the details regarding...?"
* "How does... work?"

## Technologies Used

* LangChain - Framework for LLM applications
* OpenAI - GPT-4o and text embeddings
* Pinecone - Vector database for similarity search
* Gradio - Web interface
* Python - Core programming language

## License
This project is licensed under the MIT License.
