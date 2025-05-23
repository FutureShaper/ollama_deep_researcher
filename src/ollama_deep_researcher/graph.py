import json
import uuid

from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool, Tool
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings
from langgraph.pregel.retry import RetryPolicy

from ollama_deep_researcher.configuration import Configuration, SearchAPI
from ollama_deep_researcher.utils import deduplicate_and_format_sources, format_sources, duckduckgo_search, strip_thinking_tokens, get_config_value
from ollama_deep_researcher.state import SummaryState, SummaryStateInput, SummaryStateOutput
from ollama_deep_researcher.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions, get_current_date, tool_selection_instructions

# Helper function to track and limit node visits
def track_node_visit(state: SummaryState, node_name: str, config: RunnableConfig) -> bool:
    """
    Track visits to a node and check if the maximum number of iterations has been reached.
    
    Args:
        state: Current graph state containing node_visits dictionary
        node_name: Name of the node being visited
        config: Configuration containing max_total_iterations setting
        
    Returns:
        Boolean indicating if the node can be visited again (False if max iterations reached)
    """
    # Initialize node_visits dict if needed
    if state.node_visits is None:
        state.node_visits = {}
    
    # Get the current configuration
    configurable = Configuration.from_runnable_config(config)
    max_iterations = configurable.max_total_iterations
    
    # Track the visit
    if node_name in state.node_visits:
        state.node_visits[node_name] += 1
    else:
        state.node_visits[node_name] = 1
    
    # Check if we've reached the maximum
    if state.node_visits[node_name] > max_iterations:
        print(f"[Iteration Limit] Max iterations ({max_iterations}) reached for node {node_name}")
        return False
    
    print(f"[Iteration Tracker] Node {node_name} visited {state.node_visits[node_name]}/{max_iterations} times")
    return True

"""
LangGraph-based Research Agent Architecture

This module implements a research agent with dynamic tool selection using LangGraph.

Architecture Overview:
---------------------
1. Core Tool Implementation Nodes:
   - web_research: Implements web search functionality
   - local_rag: Implements local retrieval augmented generation

2. Tool Execution Path Nodes:
   - web_search_only: Wrapper that calls web_research and handles state management
   - local_rag_only: Wrapper that calls local_rag and handles state management
   - both_tools: Wrapper that calls both tools in sequence and merges results

3. Decision Nodes:
   - decide_tool_usage: Selects which tools to use based on query analysis
   - route_to_tools: Routes execution to the appropriate tool execution path
   - check_tool_results: Evaluates if results are satisfactory or need different tools
   - route_research: Decides whether to continue research or finalize

4. Processing Nodes:
   - generate_query: Creates initial search query
   - summarize_sources: Combines research results
   - reflect_on_summary: Identifies knowledge gaps and creates follow-up queries
   - finalize_summary: Creates final output

Flow:
-----
START → generate_query → decide_tool_usage → [tool path based on selection] → 
check_tool_results → [continue or retry tools] → summarize_sources → 
reflect_on_summary → [continue research or finalize] → END

Graph Visualization Note:
------------------------
When visualizing this graph, the core implementation nodes (web_research, local_rag) may appear 
as disconnected components since they are not directly connected in the graph flow. Instead, they 
are called programmatically by the wrapper nodes (web_search_only, local_rag_only, both_tools).

This "callable node" pattern is intentional - it allows us to:
1. Keep the graph structure clean and focused on execution flow
2. Reuse the same implementation components in different execution paths
3. Maintain proper state management through the wrapper nodes

Advantages of this architecture:
------------------------------
1. Separation of concerns: Core implementations separate from execution flow
2. Dynamic tool selection: Uses vector search and LLM reasoning to choose tools
3. Feedback loops: Results are evaluated for quality before proceeding
4. Flexible state management: Each execution path properly maintains state
"""

# Define structured tools
def create_research_tools():
    """Create structured tools for research with proper metadata."""
    
    def web_search_tool(query: str) -> str:
        """Search the web for the latest information on the query."""
        return "Web search function - will be called by the graph"
    
    def local_rag_tool(query: str) -> str:
        """Search local knowledge base for relevant information about the query."""
        return "Local RAG function - will be called by the graph"
    
    # Create structured tools with proper metadata
    web_search = StructuredTool.from_function(
        web_search_tool,
        name="web_search",
        description="Search the web for recent, up-to-date information on a topic. Best for current events, trending topics, or recent developments."
    )
    
    local_rag = StructuredTool.from_function(
        local_rag_tool,
        name="local_rag",
        description="Search a local knowledge base for technical, specialized, or domain-specific information. Best for established concepts, technical details, or historical information."
    )
    
    return [web_search, local_rag]

# Create tool registry with UUIDs as keys
tools = create_research_tools()
tool_registry = {str(uuid.uuid4()): tool for tool in tools}

# Create documents for vector search
tool_documents = [
    Document(
        page_content=tool.description,
        metadata={"name": tool.name, "description": tool.description},
        id=id,
    ) 
    for id, tool in tool_registry.items()
]

# Nodes
def generate_query(state: SummaryState, config: RunnableConfig):
    print("\n[LangGraph] Entering node: generate_query")
    print(f"[LangGraph] Current state: {{'research_topic': {getattr(state, 'research_topic', None)}, 'search_query': {getattr(state, 'search_query', None)}}}")
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

def decide_tool_usage(state: SummaryState, config: RunnableConfig):
    print("\n[LangGraph] Entering node: decide_tool_usage")
    print(f"[LangGraph] Current state: {{'search_query': {getattr(state, 'search_query', None)}, 'selected_tools': {getattr(state, 'selected_tools', None)}}}")
    """LangGraph node that decides which research tools to use based on the query.
    
    Uses a hybrid approach combining vector similarity search and LLM reasoning to select
    the appropriate tools for the current query.
    
    Args:
        state: Current graph state containing the search query
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including selected_tools key with list of tools to use
    """
    configurable = Configuration.from_runnable_config(config)
    
    # First approach: Use vector search to find relevant tools
    embedding_function = OllamaEmbeddings(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm
    )
    
    # Create or get vector store
    vector_store = InMemoryVectorStore(embedding=embedding_function)
    vector_store.add_documents(tool_documents)
    
    # Perform similarity search based on the query
    results = vector_store.similarity_search(state.search_query, k=2)
    vector_selected_tools = [doc.metadata["name"] for doc in results]
    
    print(f"[Vector Selection] Selected tools: {vector_selected_tools}")
    
    # Second approach: Use LLM reasoning as a fallback or to refine the selection
    if not vector_selected_tools or len(set(vector_selected_tools)) < 2:
        # If vector search didn't return useful results, use LLM reasoning
        formatted_prompt = tool_selection_instructions.format(search_query=state.search_query)
        
        # Default to Ollama
        llm_json_mode = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            format="json"
        )
        
        result = llm_json_mode.invoke(
            [SystemMessage(content=formatted_prompt),
            HumanMessage(content=f"Decide which research tools to use for the query: '{state.search_query}'")]
        )
        
        # Parse the result
        try:
            tool_decision = json.loads(result.content)
            selected_tools = tool_decision.get('tool_selection', [])
            rationale = tool_decision.get('rationale', "No rationale provided")
            
            # Validate and ensure we have at least one tool
            if selected_tools and isinstance(selected_tools, list):
                print(f"[LLM Selection] Selected tools: {selected_tools}")
                print(f"[LLM Selection] Rationale: {rationale}")
                return {"selected_tools": selected_tools, "tool_rationale": rationale}
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    
    # Use vector search results if LLM reasoning failed or wasn't needed
    return {"selected_tools": vector_selected_tools, "tool_rationale": "Selected based on query similarity to tool descriptions"}

def web_research(state: SummaryState, config: RunnableConfig):
    print("\n[LangGraph] Entering node: web_research")
    print(f"[LangGraph] Current state: {{'search_query': {getattr(state, 'search_query', None)}, 'research_loop_count': {getattr(state, 'research_loop_count', None)}}}")
    """Core implementation node for web search functionality.
    
    This is a CORE IMPLEMENTATION NODE that is not directly connected in the graph structure.
    Instead, it is called by wrapper nodes (web_search_only, both_tools) that handle the graph flow.
    
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
    print("\n[LangGraph] Entering node: local_rag")
    print(f"[LangGraph] Current state: {{'search_query': {getattr(state, 'search_query', None)}}}")
    """Core implementation node for local RAG functionality.
    
    This is a CORE IMPLEMENTATION NODE that is not directly connected in the graph structure.
    Instead, it is called by wrapper nodes (local_rag_only, both_tools) that handle the graph flow.
    
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
    print("\n[LangGraph] Entering node: summarize_sources")
    print(f"[LangGraph] Current state: {{'running_summary': {getattr(state, 'running_summary', None)}, 'web_research_results': {getattr(state, 'web_research_results', None)}, 'local_rag_results': {getattr(state, 'local_rag_results', None)}}}")
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
    print("\n[LangGraph] Entering node: reflect_on_summary")
    print(f"[LangGraph] Current state: {{'running_summary': {getattr(state, 'running_summary', None)}}}")
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
    print("\n[LangGraph] Entering node: finalize_summary")
    print(f"[LangGraph] Current state: {{'running_summary': {getattr(state, 'running_summary', None)}, 'sources_gathered': {getattr(state, 'sources_gathered', None)}}}")
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

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "decide_tool_usage"]:
    print("\n[LangGraph] Entering node: route_research")
    print(f"[LangGraph] Current state: {{'node_visits': {getattr(state, 'node_visits', None)}}}")
    """LangGraph routing function that determines the next step in the research flow.
    
    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of iterations.
    
    Args:
        state: Current graph state containing node visit counts
        config: Configuration for the runnable, including max_total_iterations setting
        
    Returns:
        String literal indicating the next node to visit ("decide_tool_usage" or "finalize_summary")
    """
    # Check if the iteration limit has been reached for decide_tool_usage
    if not track_node_visit(state, "decide_tool_usage", config):
        print("[Research Router] Max iterations reached, finalizing summary")
        return "finalize_summary"
    
    # Continue research since we haven't hit the iteration limit
    configurable = Configuration.from_runnable_config(config)
    print(f"[Research Router] Continuing research (iteration {state.node_visits.get('decide_tool_usage', 1)}/{configurable.max_total_iterations})")
    return "decide_tool_usage"

def route_to_tools(state: SummaryState) -> Literal["web_search_only", "local_rag_only", "both_tools"]:
    print("\n[LangGraph] Entering node: route_to_tools")
    print(f"[LangGraph] Current state: {{'selected_tools': {getattr(state, 'selected_tools', None)}}}")
    """LangGraph routing function that determines which tool(s) to use.
    
    Routes the flow based on the tools selected in the decide_tool_usage node.
    
    Args:
        state: Current graph state containing the selected tools
        
    Returns:
        String literal indicating the next node to visit based on tool selection
    """
    if "web_search" in state.selected_tools and "local_rag" in state.selected_tools:
        print("[Tool Router] Using both web search and local RAG")
        return "both_tools"
    elif "web_search" in state.selected_tools:
        print("[Tool Router] Using only web search")
        return "web_search_only"
    elif "local_rag" in state.selected_tools:
        print("[Tool Router] Using only local RAG")
        return "local_rag_only"
    else:
        # Default fallback to both tools if somehow no tools were selected
        print("[Tool Router] No tools selected, defaulting to both")
        return "both_tools"

def execute_tools(state: SummaryState, config: RunnableConfig):
    print("\n[LangGraph] Entering node: execute_tools")
    print(f"[LangGraph] Current state: {{'selected_tools': {getattr(state, 'selected_tools', None)}, 'research_loop_count': {getattr(state, 'research_loop_count', None)}}}")
    """Execute all selected tools dynamically based on state.selected_tools.
    
    This unified node replaces the web_search_only, local_rag_only, and both_tools pattern
    with a more flexible approach that:
    1. Loops through all selected tools and executes each one
    2. Merges all results into a consistent state structure
    3. Handles proper state management for tools that weren't selected
    
    Args:
        state: Current graph state containing the selected tools
        config: Configuration for the runnable, including provider settings
        
    Returns:
        Dictionary with state update, including combined results from all executed tools
    """
    # Initialize a results container with default empty values
    results = {
        "web_research_results": [],
        "local_rag_results": [],
        "sources_gathered": [],
        "research_loop_count": state.research_loop_count + 1
    }
    
    # Execute each selected tool and collect results
    for tool_name in state.selected_tools:
        if tool_name == "web_search":
            print(f"[execute_tools] Executing web search...")
            web_result = web_research(state, config)
            # Merge results
            results["web_research_results"].extend(web_result.get("web_research_results", []))
            results["sources_gathered"].extend(web_result.get("sources_gathered", []))
        
        elif tool_name == "local_rag":
            print(f"[execute_tools] Executing local RAG...")
            rag_result = local_rag(state, config)
            # Merge results
            results["local_rag_results"].extend(rag_result.get("local_rag_results", []))
            
        # Future tools can be added here with simple elif statements
        # elif tool_name == "new_tool":
        #     new_tool_result = new_tool(state, config)
        #     results["new_tool_results"] = new_tool_result.get("new_tool_results", [])
    
    print(f"[execute_tools] Retrieved {len(results.get('web_research_results', []))} web results "
          f"and {len(results.get('local_rag_results', []))} local documents")
    
    return results

# Add feedback loop function to check if results are satisfactory
def check_tool_results(state: SummaryState, config: RunnableConfig) -> Literal["continue", "retry_tools"]:
    print("\n[LangGraph] Entering node: check_tool_results")
    print(f"[LangGraph] Current state: {{'web_research_results': {getattr(state, 'web_research_results', None)}, 'local_rag_results': {getattr(state, 'local_rag_results', None)}}}")
    """Check if the tool execution results are satisfactory or need different tools.
    
    Uses an LLM to evaluate if the current tool results are satisfactory or if
    we should try with different tools.
    
    Args:
        state: Current graph state containing research results
        config: Configuration for the runnable
        
    Returns:
        Decision to continue with summarization or retry with different tools
    """
    # Enforce global research loop limit ONLY
    configurable = Configuration.from_runnable_config(config)
    max_iterations = configurable.max_total_iterations
    current_loop = getattr(state, 'research_loop_count', 0)
    if current_loop >= max_iterations:
        print(f"[Result Check] Max total research iterations ({max_iterations}) reached, proceeding with current results")
        return "continue"
    
    # Get the most recent research results
    web_results = state.web_research_results[-1] if state.web_research_results else "No web research results."
    rag_results = state.local_rag_results[-1] if state.local_rag_results else "No local RAG results."
    
    # Format results for evaluation
    tools_used = ", ".join(state.selected_tools) if state.selected_tools else "No tools"
    
    # Create prompt for the LLM to evaluate results
    evaluation_prompt = f"""
    You are evaluating research results to determine if they are satisfactory or if different tools should be used.
    
    QUERY: {state.search_query}
    
    TOOLS USED: {tools_used}
    
    RESULTS SUMMARY:
    - Web Research: {web_results[:300]}... (truncated)
    - Local RAG: {rag_results[:300]}... (truncated)
    
    Evaluate if these results are likely to be sufficient and relevant for the query.
    If you believe different tools should be tried, respond with "retry_tools".
    Otherwise, respond with "continue".
    
    Your response should be ONLY one of these two options: "continue" or "retry_tools"
    """
    
    # Invoke the LLM
    llm = ChatOllama(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm,
        temperature=0
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content="You are an expert research assistant evaluating search results."),
            HumanMessage(content=evaluation_prompt)
        ])
        
        result_text = response.content.strip().lower()
        
        # Simple check for the decision
        if "retry" in result_text:
            print("[Result Check] Results not satisfactory, retrying with different tools")
            return "retry_tools"
        else:
            print("[Result Check] Results satisfactory, continuing to summarization")
            return "continue"
    except Exception as e:
        print(f"[Result Check] Error evaluating results: {e}")
        # Default to continue on error
        return "continue"

# Build the graph
# -----------------------------------------------------------------------------
# Note on the architecture:
# - Core implementation nodes: web_research, local_rag
#   These contain the actual implementation of research tools but are not directly connected in the graph
#
# - Execution path nodes: execute_tools
#   This unified node dynamically calls the core implementation nodes based on selected tools
#
# This design allows for flexible tool selection and clean state management
#
# Architecture visualization:
#
#                                 +-------------+     +------------+
#                                 | web_research|     | local_rag  |
#                                 +------+------+     +------+-----+
#                                        ^                  ^
#                                        |                  |
#                                        | calls            | calls
#                                        |                  |
#            +----------------+    +-----+-------+    +-----+------+
#   START--->|generate_query  |--->|decide_tool  |--->|execute_tools|
#            +----------------+    |   usage     |    +------+------+
#                                  +------+------+           |
#                                         |                  v 
#                                         |        +------------------+
#                                         +------->|check_tool_results|
#                                                  +--------+---------+
#                                                           |
#                                                  +--------v---------+
#                                                  |summarize_sources |
#                                                  +------------------+
# -----------------------------------------------------------------------------

# Create the graph builder
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)

# Add process and decision nodes
builder.add_node("generate_query", generate_query)  # Entry point node
builder.add_node("decide_tool_usage", decide_tool_usage, retry=RetryPolicy(max_attempts=3))  # Tool selection node
builder.add_node("summarize_sources", summarize_sources)  # Results processing node
builder.add_node("reflect_on_summary", reflect_on_summary)  # Knowledge gap analysis node
builder.add_node("finalize_summary", finalize_summary)  # Output preparation node

# Add core implementation nodes (not directly connected in the graph)
# Note: These nodes will appear as disconnected components in graph visualizations,
# as they are called programmatically by the execute_tools node
builder.add_node("web_research", web_research)  # Core web search implementation
builder.add_node("local_rag", local_rag)  # Core local RAG implementation

# Add unified tool execution node
builder.add_node("execute_tools", execute_tools)  # Dynamic tool execution node

# Connect initial process flow
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "decide_tool_usage")

# Connect tool selection directly to unified executor
builder.add_edge("decide_tool_usage", "execute_tools")

# Connect tool execution to result checking
builder.add_conditional_edges(
    "execute_tools",
    check_tool_results,
    {
        "continue": "summarize_sources",
        "retry_tools": "decide_tool_usage"
    }
)

builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

graph = builder.compile()