# AI Agent Notes

## Overview
This project explores the development of an **AI Agent** capable of reasoning, using tools, and interacting with external systems. The implementation uses **LangChain**, **MCP (Model Context Protocol)**, and **Firecrawl** for web data extraction.

The agent can analyze user queries, determine whether external information is required, and automatically use tools to retrieve relevant data.

---

## Key Technologies

### LangChain
LangChain is a framework for building applications powered by Large Language Models (LLMs). It provides abstractions for:

- LLM integrations
- Agent creation
- Tool usage
- Prompt management
- Memory handling
Eg : Imagine an LLM as a very smart brain.
LangChain provides the body and tools for that brain.
LLM (brain)+ LangChain (tools + connectors)= AI Application
- 
In this project, LangChain is used to create an agent that can decide when to call external tools.

---

### LangGraph
LangGraph is an advanced orchestration framework built on top of LangChain. It enables developers to build **stateful, multi-step AI workflows** using graph-based execution models.
A graph is made of:
- Nodes (tasks or operations)
- Edges (connections between tasks)

1. Nodes

-Nodes represent operations performed by the agent.

Examples of nodes:
- LLM reasoning
- tool execution
- memory update
- validation logic

Example:
```python
def reasoning_node(state):
    response = model.invoke(state["messages"])
    return {"messages": response}
```
The node processes input and returns updated state.

2. Edges

Edges define how the workflow moves between nodes.

Example:Reasoning Node → Tool Node → Output Node

3. State

State represents the shared data passed through the workflow.

Example state:
```python
state = {
    "messages": [],
    "tool_results": [],
    "context": {}
}
```
**LangGraph Workflow**
User Input
 ↓
Reasoning Node
 ↓
Decision Node
 ↙       ↘
Tool Node   Direct Answer
 ↓
Observation Node
 ↓
Final Response

Note:LangGraph commonly implements the ReAct pattern:Thought → Action → Observation → Answer

Example:
Thought: Need current gold price
Action: Scrape website
Observation: Gold price = $2312
Answer: Current price is $2312


---

### Tools
Tools extend the capabilities of an LLM by allowing it to interact with external systems.

Examples of tools include:

- Web scraping
- API calls
- Database queries
- Search engines

In this project, tools are provided by **Firecrawl via MCP**, enabling the agent to scrape and extract information from websites.

---

### MCP (Model Context Protocol)
MCP is a standardized protocol that allows AI models to communicate with external tools and services.

It simplifies tool integration by providing a structured interface between AI agents and external capabilities.

Architecture:
Agent → MCP Client → MCP Server → External Tool

This project connects to the **Firecrawl MCP server**, which exposes web scraping tools that the agent can call dynamically.

---

## Agent Workflow

Typical execution flow:User Query  → Agent Reasoning → Determine if Tool is Required → Call MCP Tool → Retrieve Data → Generate Final Response

