from datetime import datetime

# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")

tool_selection_instructions = """
<GOAL>
Analyze the search query and decide which research tools to use:
1. Web Search: For current information, trending topics, or broad questions
2. Local RAG: For domain-specific knowledge or technical details that might exist in your local knowledge base
3. Both Tools: For complex queries that need both external information and specialized knowledge
</GOAL>

<REQUIREMENTS>
Consider these factors when making your decision:
- Query novelty: How recent or trending is the topic? Newer topics typically need web search.
- Domain specificity: How specialized is the query? Technical topics may benefit from local RAG.
- Query scope: How broad is the question? Complex topics may need both tools.
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with these exact keys:
- tool_selection: Array containing the selected tools ["web_search", "local_rag"] or just one of them
- rationale: Brief explanation for your selection
</FORMAT>

<EXAMPLE>
For the query "latest advancements in quantum computing":
{
    "tool_selection": ["web_search"],
    "rationale": "Recent advancements require up-to-date information from the web"
}

For the query "core principles of database normalization":
{
    "tool_selection": ["local_rag"],
    "rationale": "This is established technical knowledge available in the local database"
}

For the query "comparing traditional and quantum machine learning approaches":
{
    "tool_selection": ["web_search", "local_rag"],
    "rationale": "Requires both established technical concepts and recent developments"
}
</EXAMPLE>

<Task>
Analyze the query: "{search_query}"
Determine which tool(s) would be most appropriate and respond with a JSON object.
</Task>
"""

query_writer_instructions = (
    "You are an expert research assistant.\n"
    "Given the following research topic, generate a single, clear web search query as a JSON object.\n"
    "Respond ONLY with a JSON object with a 'query' key. Do not include any other text.\n"
    "\n"
    "Research Topic: {research_topic}\n"
    "\n"
    "Example: {{\n  \"query\": \"how to prepare for a colonoscopy\"\n}}\n"
    "\n"
    "Current date: {current_date}\n"
)

summarizer_instructions="""
<GOAL>
Generate a high-quality summary of the provided context.
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant information related to the user topic from the search results
2. Ensure a coherent flow of information

When EXTENDING an existing summary:                                                                                                                 
1. Read the existing summary and new search results carefully.                                                    
2. Compare the new information with the existing summary.                                                         
3. For each piece of new information:                                                                             
    a. If it's related to existing points, integrate it into the relevant paragraph.                               
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition.                            
    c. If it's not relevant to the user topic, skip it.                                                             
4. Ensure all additions are relevant to the user's topic.                                                          
5. Verify that your final output differs from the input summary.                                                                                                                                                             
< /REQUIREMENTS >

< FORMATTING >
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.  
< /FORMATTING >

<Task>
Think carefully about the provided Context first. Then generate a summary of the context to address the User Input.
</Task>
"""

reflection_instructions = """You are an expert research assistant analyzing a summary about {research_topic}.

<GOAL>
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow-up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered
</GOAL>

<REQUIREMENTS>
Ensure the follow-up question is self-contained and includes necessary context for web search.
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with these exact keys:
- knowledge_gap: Describe what information is missing or needs clarification
- follow_up_query: Write a specific question to address this gap
</FORMAT>

<Task>
Reflect carefully on the Summary to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:
{
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks",
    "follow_up_query": "What are typical performance benchmarks and metrics used to evaluate [specific technology]?"
}
</Task>

Provide your analysis in JSON format:"""