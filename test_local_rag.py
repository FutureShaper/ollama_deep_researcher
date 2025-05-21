"""
Test script for local RAG node without running the full LangGraph server.
This script simulates the state and config, and directly calls the local_rag function.
"""
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from ollama_deep_researcher.configuration import Configuration
from ollama_deep_researcher.graph import generate_query, local_rag
from ollama_deep_researcher.state import SummaryState

# Simulate a config (use your real config if needed)
class DummyConfig:
    def __init__(self):
        self.ollama_base_url = "http://localhost:11434"
        self.local_llm = "qwen3:14b"
    def get(self, name, default=None):
        return getattr(self, name, default)

# Simulate a state with a search_query
state = SummaryState(
    research_topic="What does effective program management in cloud software require?",
    search_query="",  # search_query is initially empty
    running_summary=None,
    web_research_results=[],
    sources_gathered=[],
    research_loop_count=0
)

# Use the real Configuration if you want to test with .env/config
config = {"configurable": DummyConfig()}

# Step 1: Generate the search_query
query_result = generate_query(state, config)
print("\ngenerate_query output:")
print(query_result)
state.search_query = query_result["search_query"]

# Step 2: Use local_rag with the generated search_query
result = local_rag(state, config)
print("\nlocal_rag output:")
print(result)
