from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from ollama_deep_researcher.configuration import Configuration

# Path to your document
DOC_PATH = "program_release_management.txt"

# Load and split document
with open(DOC_PATH, "r") as f:
    text = f.read()

# Simple split by paragraphs (double newlines)
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

# Prepare documents
docs = [Document(page_content=p) for p in paragraphs]

# Set up embedding function and Chroma vector store
configurable = Configuration()
embedding_function = OllamaEmbeddings(
    base_url=configurable.ollama_base_url,
    #model=configurable.local_llm
    model="qwen3:14b"
)
persist_directory = "./chroma_langchain_db"
collection_name = "deep_research_collection"
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_function,
    persist_directory=persist_directory
)

# Add documents to Chroma
vector_store.add_documents(docs)
print(f"Added {len(docs)} paragraphs to ChromaDB from {DOC_PATH}.")
