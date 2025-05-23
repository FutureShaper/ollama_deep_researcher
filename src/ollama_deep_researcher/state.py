import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated

@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None) # Report topic     
    search_query: str = field(default=None) # Search query
    web_research_results: Annotated[list, operator.add] = field(default_factory=list) 
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list) 
    research_loop_count: int = field(default=0) # Research loop count
    running_summary: str = field(default=None) # Final report
    local_rag_results: Annotated[list, operator.add] = field(default_factory=list)  # Add this line for UI visibility
    selected_tools: list = field(default_factory=list)  # Track which tools to use for the current query
    node_visits: dict = field(default_factory=dict)  # Track number of visits per node

@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str  # Report topic (required)

@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None) # Final report
    local_rag_results: list = field(default_factory=list)  # Add for output schema