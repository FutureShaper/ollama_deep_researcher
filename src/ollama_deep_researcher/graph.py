import json

from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings

from ollama_deep_researcher.configuration import Configuration, SearchAPI
from ollama_deep_researcher.utils import deduplicate_and_format_sources, format_sources, duckduckgo_search, strip_thinking_tokens, get_config_value
from ollama_deep_researcher.state import SummaryState, SummaryStateInput, SummaryStateOutput
from ollama_deep_researcher.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions, get_current_date

# Nodes
def generate_query(state: SummaryState, config: RunnableConfig):
    """LangGraph node that generates a search query based on the research topic.
    
    Uses an LLM to create an optimized search query for web research based on
    the user's research topic. Defaults to Ollama as LLM provider.
    
    Args:
        state: Current graph state containing the research topic
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """

    # Ensure research_topic is set
    if not state.research_topic or not state.research_topic.strip():
        raise ValueError("research_topic must be provided and non-empty for query generation.")

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=state.research_topic
    )

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    
    # Default to Ollama
    llm_json_mode = ChatOllama(
        base_url=configurable.ollama_base_url, 
        model=configurable.local_llm, 
        temperature=0, 
        format="json"
    )
    
    result = llm_json_mode.invoke(
        [SystemMessage(content=formatted_prompt),
        HumanMessage(content=f"Generate a query for web search:")]
    )
    
    # Get the content
    content = result.content

    # Debug: return the raw LLM output for debugging
    try:
        query = json.loads(content)
        search_query = query.get('query')
        # Fallback: if query is None, try to extract a string from the whole response
        if not search_query or search_query in ('{}', None, "null"):
            # Final fallback: use the research topic itself
            search_query = state.research_topic
    except (json.JSONDecodeError, KeyError, ValueError):
        # If parsing fails or the key is not found, use a fallback query
        if configurable.strip_thinking_tokens:
            content = strip_thinking_tokens(content)
        search_query = state.research_topic  # Final fallback
    return {"search_query": search_query, "llm_raw_output": content}

def web_research(state: SummaryState, config: RunnableConfig):
    """LangGraph node that performs web research using the generated search query.
    
    Executes a web search using the configured search API (tavily, perplexity, 
    duckduckgo, or searxng) and formats the results for further processing.
    
    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings
        
    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Get the search API
    search_api = get_config_value(configurable.search_api)

    # Search the web
    if search_api == "tavily":
        search_results = tavily_search(state.search_query, fetch_full_page=configurable.fetch_full_page, max_results=1)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "perplexity":
        search_results = perplexity_search(state.search_query, state.research_loop_count)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "duckduckgo":
        search_results = duckduckgo_search(state.search_query, max_results=3, fetch_full_page=configurable.fetch_full_page)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "searxng":
        search_results = searxng_search(state.search_query, max_results=3, fetch_full_page=configurable.fetch_full_page)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, fetch_full_page=configurable.fetch_full_page)
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {"sources_gathered": [format_sources(search_results)], "research_loop_count": state.research_loop_count + 1, "web_research_results": [search_str]}

def local_rag(state: SummaryState, config: RunnableConfig):
    """LangGraph node that performs local RAG using ChromaDB.
    Retrieves relevant documents from a local Chroma vector store based on the current search query.
    Returns a list of retrieved documents as 'local_rag_results'.
    """
    # Set up embedding function (adjust as needed)
    configurable = Configuration.from_runnable_config(config)
    embedding_function = OllamaEmbeddings(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm
    )
    # Initialize Chroma vector store (persisted locally)
    persist_directory = "./chroma_langchain_db"
    collection_name = "deep_research_collection"
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )
    # Perform similarity search
    query = state.search_query
    print(f"[local_rag] search_query: {query}")
    results = vector_store.similarity_search(query, k=3)
    print(f"[local_rag] Retrieved {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content}")
    # Format results for downstream use
    rag_texts = [doc.page_content for doc in results]
    # Persist results in state for UI visibility
    state.local_rag_results = rag_texts
    return {"local_rag_results": rag_texts}

def summarize_sources(state: SummaryState, config: RunnableConfig):
    """LangGraph node that summarizes web research and local RAG results.
    Uses an LLM to create or update a running summary based on the newest web and local RAG results.
    """
    # Combine web and local RAG results for summarization
    combined_context = []
    if state.web_research_results:
        combined_context.extend(state.web_research_results)
    if state.local_rag_results:
        combined_context.extend(state.local_rag_results)
    if not combined_context:
        combined_context = ["No context available."]

    existing_summary = state.running_summary
    most_recent_context = combined_context[-1]

    if existing_summary:
        human_message_content = (
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Context> \n {most_recent_context} \n <New Context>"
            f"Update the Existing Summary with the New Context on this topic: \n <User Input> \n {state.research_topic} \n <User Input>\n\n"
        )
    else:
        human_message_content = (
            f"<Context> \n {most_recent_context} \n <Context>"
            f"Create a Summary using the Context on this topic: \n <User Input> \n {state.research_topic} \n <User Input>\n\n"
        )

    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(
        base_url=configurable.ollama_base_url, 
        model=configurable.local_llm, 
        temperature=0
    )
    result = llm.invoke([
        SystemMessage(content=summarizer_instructions),
        HumanMessage(content=human_message_content)
    ])
    running_summary = result.content
    if configurable.strip_thinking_tokens:
        running_summary = strip_thinking_tokens(running_summary)
    return {"running_summary": running_summary}

def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """LangGraph node that identifies knowledge gaps and generates follow-up queries.
    
    Analyzes the current summary to identify areas for further research and generates
    a new search query to address those gaps. Uses structured output to extract
    the follow-up query in JSON format.
    
    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    
    # Default to Ollama
    llm_json_mode = ChatOllama(
        base_url=configurable.ollama_base_url, 
        model=configurable.local_llm, 
        temperature=0, 
        format="json"
    )
    
    result = llm_json_mode.invoke(
        [SystemMessage(content=reflection_instructions.format(research_topic=state.research_topic)),
        HumanMessage(content=f"Reflect on our existing knowledge: \n === \n {state.running_summary}, \n === \n And now identify a knowledge gap and generate a follow-up web search query:")]
    )
    
    # Strip thinking tokens if configured
    try:
        # Try to parse as JSON first
        reflection_content = json.loads(result.content)
        # Get the follow-up query
        query = reflection_content.get('follow_up_query')
        # Check if query is None or empty
        if not query:
            # Use a fallback query
            return {"search_query": f"Tell me more about {state.research_topic}"}
        return {"search_query": query}
    except (json.JSONDecodeError, KeyError, AttributeError):
        # If parsing fails or the key is not found, use a fallback query
        return {"search_query": f"Tell me more about {state.research_topic}"}
        
def finalize_summary(state: SummaryState):
    """LangGraph node that finalizes the research summary.
    
    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.
    
    Args:
        state: Current graph state containing the running summary and sources gathered
        
    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """

    # Deduplicate sources before joining
    seen_sources = set()
    unique_sources = []
    
    for source in state.sources_gathered:
        # Split the source into lines and process each individually
        for line in source.split('\n'):
            # Only process non-empty lines
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append(line)
    
    # Join the deduplicated sources
    all_sources = "\n".join(unique_sources)
    state.running_summary = f"## Summary\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    """LangGraph routing function that determines the next step in the research flow.
    
    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.
    
    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_web_research_loops setting
        
    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary"

# Add nodes and edges
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("local_rag", local_rag)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Add edges
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "local_rag")
builder.add_edge("local_rag", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

graph = builder.compile()