"""
Run the full LangGraph research agent from a Python script.
"""
from ollama_deep_researcher.graph import graph
from ollama_deep_researcher.state import SummaryStateInput

# Set your research topic here
RESEARCH_TOPIC = "What does effective program management in cloud software require?"

# Prepare the input state
input_state = SummaryStateInput(research_topic=RESEARCH_TOPIC)

# Run the graph (synchronously)
result = graph.invoke(input_state)

print("\n===== FINAL OUTPUT =====")
print(result)
