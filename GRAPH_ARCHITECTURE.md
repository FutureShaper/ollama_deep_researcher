# LangGraph Architecture Guide

## Dynamic Tool Selection Architecture

This project uses a scalable pattern for dynamic tool selection in LangGraph that allows for flexible tool composition without requiring structure changes to the graph.

### Core Components

1. **Core Implementation Nodes**
   - `web_research`: Contains web search functionality 
   - `local_rag`: Contains local RAG functionality
   - These nodes are *not directly connected* in the graph flow

2. **Unified Tool Execution**
   - `execute_tools`: A single, unified execution node that dynamically calls the appropriate core implementation nodes based on what's in `state.selected_tools`
   - Handles all possible tool combinations without requiring separate paths for each combination

### Graph Visualization

When viewing the graph visualization (e.g., in LangSmith or via the langgraph dev UI), you'll notice:

- The core implementation nodes (`web_research`, `local_rag`) appear as disconnected components
- The actual flow goes through the unified `execute_tools` node which programmatically calls the core implementation nodes

This is by design and allows us to:
1. Create a cleaner, more maintainable graph structure
2. Dynamically execute only the selected tools without hardcoding execution paths
3. Scale to additional tools without requiring graph structure changes
4. Maintain proper state management in all scenarios

### Flow Logic

1. Query Generation: `generate_query` creates a search query
2. Tool Selection: `decide_tool_usage` decides which tools to use based on the query and populates `state.selected_tools`
3. Execution: `execute_tools` dynamically runs each tool in `state.selected_tools`
4. Result Checking: `check_tool_results` evaluates if the results are satisfactory
5. Processing: `summarize_sources` combines the results
6. Reflection: `reflect_on_summary` identifies knowledge gaps
7. Next Cycle: Either continue research with new tools or finalize

### Adding New Tools

To add a new tool to the system:

1. Create a new core implementation node for the tool:
   ```python
   def new_tool(state: SummaryState, config: RunnableConfig):
       # Implementation
       return {"new_tool_results": [result]}
   ```

2. Register the tool in the `create_research_tools` function:
   ```python
   new_tool = StructuredTool.from_function(
       new_tool_function,
       name="new_tool",
       description="Description of what the new tool does."
   )
   ```

3. Update the `execute_tools` function to handle the new tool:
   ```python
   elif tool_name == "new_tool":
       new_tool_result = new_tool(state, config)
       results["new_tool_results"] = new_tool_result.get("new_tool_results", [])
   ```

4. Update `SummaryState` in state.py to include the new results:
   ```python
   new_tool_results: Annotated[list, operator.add] = field(default_factory=list)
   ```

That's it! No need to modify the graph structure or create new wrapper nodes.

This architecture pattern provides clean separation of concerns, flexible tool selection, and excellent scalability.
