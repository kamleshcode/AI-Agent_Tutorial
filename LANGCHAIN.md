# LangChain Complete Guide
> From Basics to Production-Ready Agents

---

## Table of Contents
1. [What is LangChain?](#1-what-is-langchain)
2. [Core Concepts](#2-core-concepts)
3. [Messages](#3-messages)
4. [Models](#4-models)
5. [Tools & Tool Binding](#5-tools--tool-binding)
6. [Agents](#6-agents)
7. [Structured Output](#7-structured-output)
8. [Streaming](#8-streaming)
9. [Memory & Checkpointing](#9-memory--checkpointing)
10. [Middleware](#10-middleware)
11. [Model Context Protocol (MCP)](#11-model-context-protocol-mcp)
12. [Quick Reference](#12-quick-reference)

---

## 1. What is LangChain?

**LangChain** is a framework for building applications powered by Large Language Models (LLMs). It provides abstractions for chaining together language models, tools, memory, and agents to build sophisticated AI-powered applications.

At its core, LangChain lets you:
- Connect LLMs to external tools and APIs
- Build agents that can reason and take multi-step actions
- Add memory so conversations persist across turns
- Stream responses for real-time output
- Structure and validate model outputs

---

## 2. Core Concepts

| Concept | Description |
|---|---|
| **Model** | The LLM reasoning engine (e.g. GPT, Claude, Gemini) |
| **Tool** | A function the model can call to perform actions |
| **Agent** | Model + Tools combined into a reasoning loop |
| **Message** | A unit of conversation (user, assistant, system, tool) |
| **Checkpointer** | Saves agent state for memory and multi-turn conversations |
| **Middleware** | Intercepts and modifies agent execution at various points |
| **MCP** | Model Context Protocol — standard for exposing tools to LLMs |

---

## 3. Messages

Messages are the fundamental objects in LangChain. Every interaction is built on them.

### Message Types

| Type | Role | Purpose |
|---|---|---|
| `SystemMessage` | `system` | Sets tone, role, and guidelines for the model |
| `HumanMessage` | `user` | Represents user input |
| `AIMessage` | `assistant` | Represents model output |
| `ToolMessage` | `tool` | Passes tool execution results back to the model |

### Example
```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

messages = [
    SystemMessage(content="You are a helpful math assistant."),
    HumanMessage(content="What is 4 x 5?"),
    AIMessage(content="The answer is 20."),
]
```

### AIMessage — Usage Metadata
You can inspect token usage and cost metadata from AI responses:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano")
response = model.invoke("Hello!")

print(response.usage_metadata)
# {
#   'input_tokens': 8,
#   'output_tokens': 304,
#   'total_tokens': 312,
#   'input_token_details': {'audio': 0, 'cache_read': 0},
#   'output_token_details': {'audio': 0, 'reasoning': 256}
# }
```

### ToolMessage — Key Attributes

| Attribute | Description |
|---|---|
| `content` | The result returned by the tool |
| `tool_call_id` | Links the result to the specific tool call |
| `name` | Name of the tool that was called |
| `artifact` | Supplementary data NOT sent to the model (accessible programmatically) |

---

## 4. Models

The model is the **reasoning engine** of your agent.

### Initializing a Model
```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano")
response = model.invoke("Tell me a joke.")
print(response.content)
```

### Static vs Dynamic Models

**Static Model** — configured once, unchanged throughout execution:
```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5-nano",   # fixed model
    tools=[my_tool],
)
```

**Dynamic Model** — selected at runtime based on context (useful for cost optimization and routing):
```python
from langchain.agents import wrap_model_call

@wrap_model_call
def dynamic_model_selector(request):
    # Switch models based on task complexity
    if "analyze" in request.messages[-1].content.lower():
        request.model = "gpt-5.4"       # expensive, powerful
    else:
        request.model = "gpt-5-nano"    # cheap, fast
    return request
```

---

## 5. Tools & Tool Binding
 
Tools are **functions the model can call** to interact with the outside world — APIs, databases, calculators, etc.
 
### Defining a Tool
 
There are two ways to define a tool:
 
#### Way 1: Plain Python Function
A regular function with a docstring. LangChain uses the **function name** as the tool name and the **docstring** as the description the model reads to decide when to use it.
 
```python
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```
 
#### Way 2: `@tool` Decorator (Recommended)
The `@tool` decorator explicitly marks a function as a LangChain tool. It gives you more control — custom name, description, and argument parsing.
 
```python
from langchain.tools import tool
 
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers. Use this when you need to calculate a product."""
    return a * b
```
 
You can also **override the name and description**:
```python
@tool(name="calculator_multiply", description="Multiplies two integers and returns the result.")
def multiply(a: int, b: int) -> int:
    return a * b
```
 
#### Why prefer `@tool`?
 
| | Plain Function | `@tool` Decorator |
|---|---|---|
| Works as a tool | ✅ | ✅ |
| Custom name/description | ❌ | ✅ |
| Explicit & readable | ❌ | ✅ |
| Can add metadata | ❌ | ✅ |
| Recommended for production | ❌ | ✅ |
 
> 💡 The **docstring is critical** — the model reads it to decide when and how to use the tool. Write it clearly and specifically.
 
### Connecting Models to Tools — Two Methods
 
#### Method 1: `bind_tools()`
Directly attaches tools to a model. The model can suggest tool calls, but you handle execution manually.
 
```python
model_with_tools = model.bind_tools([multiply])
 
response = model_with_tools.invoke("Multiply 4 and 5")
print(response.tool_calls)
# [{'name': 'multiply', 'args': {'a': 4, 'b': 5}, 'id': '...'}]
```
 
#### Method 2: `create_agent()`
Creates a full agent loop — the model reasons, calls tools, and handles the results automatically.
 
```python
from langchain.agents import create_agent
 
agent = create_agent(
    model=model,
    tools=[multiply],
    system_prompt="You are a math assistant."
)
 
response = agent.invoke({
    "messages": [{"role": "user", "content": "Multiply 8 and 6"}]
})
 
for msg in response["messages"]:
    print(msg.type.upper(), ":", msg.content)
```
 
### When to Use Which?
 
| | `bind_tools()` | `create_agent()` |
|---|---|---|
| Tool loop handled automatically | ❌ | ✅ |
| Full reasoning + multi-step | ❌ | ✅ |
| Manual control over execution | ✅ | ❌ |
| Best for | Simple integrations | Production agents |
 
---
## 6. Agents

An **agent** combines a language model with tools to form a loop that can:
1. Receive a task
2. Reason about what to do
3. Call tools as needed
4. Return a final answer

### Creating an Agent
```python
from langchain.agents import create_agent

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[search_tool, calculator_tool],
    system_prompt="You are a research assistant."
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "What is 12% of 850?"}]
})
```

### Invocation Methods

| Method | Description |
|---|---|
| `.invoke()` | Synchronous call, waits for full response |
| `.ainvoke()` | Async version of invoke |
| `.stream()` | Synchronous streaming, yields chunks |
| `.astream()` | Async streaming |

---

## 7. Structured Output

Force the model to return data in a specific schema using **Pydantic models**.

### Strategy 1: `ToolStrategy`
Uses artificial tool calling under the hood to generate structured output.

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-5.4-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({"messages": [{"role": "user", "content": "Find John's contact info."}]})
# result will be a ContactInfo object
```

### Strategy 2: `ProviderStrategy`
Uses the model provider's **native** structured output feature — generally more reliable.

```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-5.4",
    response_format=ProviderStrategy(ContactInfo)
)
```

### Comparison

| | `ToolStrategy` | `ProviderStrategy` |
|---|---|---|
| Works with all models | ✅ | ❌ (provider must support it) |
| Uses native API feature | ❌ | ✅ |
| Reliability | Good | Better |

> ⚠️ **Gotcha with Checkpointers:** If using `response_format` alongside a checkpointer, a failed schema validation in turn N may save a stale state. Use isolated subgraphs or clear state-flushing patterns to avoid this.

---

## 8. Streaming

Streaming lets you receive and display output **as it's generated**, rather than waiting for the full response.

### Basic Streaming
```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Tell me about space."}]}
):
    print(chunk)
```

### Stream Modes

| Mode | Description |
|---|---|
| `updates` | Streams state updates after each agent step |
| `messages` | Streams `(token, metadata)` tuples from LLM nodes |
| `custom` | Streams custom data you write from inside tools |

1. ```updates``` — Step-by-step state changes
- Streams the state of the agent after each step completes. You see what changed at every node (model call, tool call, etc.).
```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in Delhi?"}]},
    stream_mode="updates"
):
    print(chunk)
```
What we get:
```python
{'agent': {'messages': [AIMessage(content='', tool_calls=[{'name': 'get_weather', ...}])]}}
{'tools': {'messages': [ToolMessage(content="It's 38°C and sunny", ...)]}}
{'agent': {'messages': [AIMessage(content='The weather in Delhi is 38°C and sunny.')]}}
```
Each chunk tells you which node ran and what it produced. Great for tracking progress.
When to use: Building a UI that shows "thinking → calling tool → got result → answering". Like a progress tracker.

2. ```messages``` — Token by token from the LLM
- Streams (token, metadata) tuples as the LLM generates them, word by word. This is the classic "ChatGPT typing effect."
```python
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "Explain quantum computing"}]},
    stream_mode="messages"
):
    print(token.content, end="", flush=True)
```
What you get:
```python
Quantum  computing  is  a  type  of  computation  that ...
```
Each chunk is a tiny piece of the model's output as it's being generated — not after it's done.
When to use: Any chat interface where you want the response to appear live as it's typed. Most consumer-facing apps use this.

3. ```custom``` — Your own events from inside tools
- Streams data you manually emit from inside tool functions using ```get_stream_writer()```.
- These updates are:
   - NOT generated by the model 
   - NOT generated automatically by the agent

They are manually emitted by the developer.
```python
from langgraph.config import get_stream_writer

@tool
def analyze_data(query: str) -> str:
    """Analyze data for a given query."""
    writer = get_stream_writer()

    writer("Step 1: Fetching data...")       # you emit this
    data = fetch_data(query)

    writer("Step 2: Running analysis...")    # you emit this
    result = run_analysis(data)

    writer("Step 3: Done!")
    return result
```

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Analyze sales data"}]},
    stream_mode="custom"
):
    if chunk["type"] == "custom":
        print(chunk["data"])   # prints your manual updates
```
What you get:
```python
Step 1: Fetching data...
Step 2: Running analysis...
Step 3: Done!
```
When to use: Long-running tools where the user needs progress feedback. Like a file upload, a web scraper, or a multi-step computation.

### Combining Multiple Modes
You can pass a list to get multiple streams at once:
```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Research AI trends"}]},
    stream_mode=["updates", "custom"]
):
    print(chunk)
```
| Mode       | What It Streams               | Best For                               |
| ---------- | ----------------------------- | -------------------------------------- |
| `updates`  | Agent state after each step   | Progress tracking, debugging, logs     |
| `messages` | LLM tokens while generating   | Chat UI, typing effect, live responses |
| `custom`   | Developer-emitted tool events | Long-running tools, progress updates   |


### Streaming from Inside Tools
Use `get_stream_writer` to emit custom events during tool execution:

```python
from langchain.agents import create_agent
from langgraph.config import get_stream_writer

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")   # streamed to client
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(model="claude-sonnet-4-6", tools=[get_weather])

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Weather in SF?"}]},
    stream_mode="custom",
    version="v2",
):
    if chunk["type"] == "custom":
        print(chunk["data"])
```

---

## 9. Memory & Checkpointing

By default, agents have **no memory** — each invocation is independent. Checkpointers add persistent state across turns using `thread_id`.

### How it Works
- The checkpointer saves the full agent state after every step
- Each conversation gets a unique `thread_id`
- On the next turn, the agent loads state from that thread

### Example — Multi-Turn Memory
```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()

agent = create_agent(
    model="gpt-5-nano",
    tools=[multiply],
    checkpointer=memory
)

# Turn 1 — User introduces themselves
agent.invoke(
    {"messages": [{"role": "user", "content": "Hi, my name is Alex."}]},
    config={"configurable": {"thread_id": "101"}}
)

# Turn 2 — Agent remembers Alex
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    config={"configurable": {"thread_id": "101"}}
)

print(response["messages"][-1].content)
# Output: "Your name is Alex."
```

### Thread Isolation
Different `thread_id` values create completely separate conversations — no state leaks between users.

```python
# User A — Thread 101
agent.invoke({"messages": [...]}, config={"configurable": {"thread_id": "101"}})

# User B — Thread 202 (separate memory)
agent.invoke({"messages": [...]}, config={"configurable": {"thread_id": "202"}})
```

---

## 10. Middleware

Middleware lets you **intercept and control** agent execution — for logging, safety, cost control, retries, and more.

### Hook Types

#### Node-Style Hooks — run at specific points
| Hook | When It Runs |
|---|---|
| `before_agent` | Once before the agent starts |
| `before_model` | Before each model call |
| `after_model` | After each model response |
| `after_agent` | Once after the agent completes |

#### Wrap-Style Hooks — wrap each call
| Hook | When It Runs |
|---|---|
| `wrap_model_call` | Around every model call |
| `wrap_tool_call` | Around every tool call |

### Built-in Middleware

| Middleware | Purpose |
|---|---|
| `SummarizationMiddleware` | Auto-summarize history when approaching token limits |
| `HumanInTheLoopMiddleware` | Pause for human approval before tool calls |
| `ModelCallLimitMiddleware` | Limit total model calls to control costs |
| `ToolCallLimitMiddleware` | Limit calls to specific tools |
| `ModelFallbackMiddleware` | Switch to backup model if primary fails |
| `PIIMiddleware` | Detect and redact/mask personal information |
| `LLMToolSelectorMiddleware` | Use an LLM to pre-filter relevant tools |
| `ToolRetryMiddleware` | Auto-retry failed tools with exponential backoff |
| `ModelRetryMiddleware` | Auto-retry failed model calls |
| `LLMToolEmulator` | Emulate tool execution via LLM (for testing) |
| `SubagentMiddleware` | Allow agents to spawn child sub-agents |

### Usage Examples

```python
from langchain.agents import create_agent
from langchain.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ModelFallbackMiddleware,
    PIIMiddleware,
    ToolRetryMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-5.4",
    tools=[my_tool],
    checkpointer=InMemorySaver(),  # required for ModelCallLimitMiddleware
    middleware=[
        # Summarize history after 4000 tokens, keep last 20 messages
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),

        # Require human approval for sensitive tools
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
                "read_email": False,  # never interrupt for this tool
            }
        ),

        # Max 10 model calls per thread, 5 per run
        ModelCallLimitMiddleware(thread_limit=10, run_limit=5, exit_behavior="end"),

        # Limit search tool to 5 calls per thread
        ToolCallLimitMiddleware(tool_name="search", thread_limit=5, run_limit=3),

        # Fallback chain if primary model fails
        ModelFallbackMiddleware("gpt-5.4-mini", "claude-3-5-sonnet-20241022"),

        # Redact PII before sending to model
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),

        # Retry failed tools up to 3 times
        ToolRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0),
    ]
)
```

---

## 11. Model Context Protocol (MCP)

**MCP** is an open standard that lets LangChain agents use tools defined on external servers — like plugins but standardized.

### Install
```bash
pip install langchain-mcp-adapters fastmcp
```

### Quickstart — Connecting to MCP Servers
```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

async def main():
    client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",           # local subprocess
            "command": "python",
            "args": ["/path/to/math_server.py"],
        },
        "weather": {
            "transport": "http",            # remote HTTP server
            "url": "http://localhost:8000/mcp",
        }
    })

    tools = await client.get_tools()
    agent = create_agent("claude-sonnet-4-6", tools)

    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is (3 + 5) x 12?"}]}
    )
    print(response)

asyncio.run(main())
```

### Creating a Custom MCP Server
```python
# math_server.py
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### Transport Types

| Transport | Use Case |
|---|---|
| `stdio` | Local subprocess, simple setup, inherently stateful |
| `http` | Remote servers, production deployments |

1. ```stdio``` — Local, runs on your machine
- Your agent launches the server itself as a subprocess and talks to it through standard input/output (like typing into a terminal).
```python
"math": {
    "transport": "stdio",
    "command": "python",              # how to start the server
    "args": ["/path/to/math_server.py"]  # which file to run
}
```
What Actually Happens:
```python
Agent → starts math_server.py as a subprocess → sends/receives data via stdin/stdout
```
2. ```http``` — Remote, runs somewhere else
- Your agent sends HTTP requests to an already-running server at a URL. The server can be anywhere — same machine, different machine, cloud, etc.
```python
"weather": {
    "transport": "http",
    "url": "http://localhost:8000/mcp"   # where the server is running
}
```
What Actually Happens:
```python
Agent → sends HTTP request to localhost:8000 → gets response back
```

### Stateless vs Stateful Sessions

By default, `MultiServerMCPClient` is **stateless** — each tool call creates a fresh session.

For **stateful** sessions (server maintains context across calls):
```python
async with client.session("server_name") as session:
    tools = await load_mcp_tools(session)
    agent = create_agent("gemini-3.1-pro-preview", tools)
```

### MCP Tool Interceptors
An interceptor is a middleman that sits between your agent and the MCP tool. Every time a tool is called, the interceptor runs first.
```python
Agent → Interceptor → MCP Tool → Interceptor → Agent
```
Interceptors let you modify, log, or gate MCP tool calls at runtime.

Basic Structure:
Every interceptor has the same shape — always two parameters, always returns a result:
```python
async def my_interceptor(request, handler):
    # request = the tool call (name, args, etc.)
    # handler = the actual tool — call this to execute it

    result = await handler(request)   # run the tool
    return result
```
Real Usage Example:
```python
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

# Logging interceptor
async def logging_interceptor(request: MCPToolCallRequest, handler):
    print(f"Calling: {request.name} with {request.args}")
    result = await handler(request)
    print(f"Result: {result}")
    return result

# Retry interceptor
async def retry_interceptor(request: MCPToolCallRequest, handler):
    for attempt in range(3):
        try:
            return await handler(request)
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2 ** attempt)

client = MultiServerMCPClient(
    {"math": {"transport": "stdio", "command": "python", "args": ["/path/to/server.py"]}},
    tool_interceptors=[logging_interceptor, retry_interceptor],
)
```

#### Interceptors Compose Like an Onion
```
outer: before → inner: before → tool execution → inner: after → outer: after
```

### Runtime Context in Interceptors

When MCP is used inside a LangChain agent, interceptors get access to:

| Property | What it gives you |
|---|---|
| `runtime.context` | User-defined context (user ID, API key, etc.) |
| `runtime.store` | Long-term memory store |
| `runtime.state` | Current agent conversation state |
| `runtime.tool_call_id` | ID of the current tool call |

```python
async def inject_user_context(request: MCPToolCallRequest, handler):
    user_id = request.runtime.context.user_id
    modified = request.override(args={**request.args, "user_id": user_id})
    return await handler(modified)
```

### MCP Core Features Summary

| Feature | Method | Returns |
|---|---|---|
| Tools | `client.get_tools()` | LangChain tool list |
| Resources | `client.get_resources("server")` | `Blob` objects |
| Prompts | `client.get_prompt("server", "name")` | LangChain messages |

1. What is a Blob?
- A Blob is LangChain's unified container for any kind of data a resource returns — text files, JSON, binary files, images, etc.

2. What's Inside a Blob?
```python
blobs = await client.get_resources("server_name")

for blob in blobs:
    print(blob.mimetype)        # what type of data  → "text/plain", "application/json", "image/png"
    print(blob.metadata["uri"]) # where it came from → "file:///path/to/file.txt"
    print(blob.as_string())     # actual content      → only works for text-based data
    print(blob.data)            # raw bytes           → works for any type including binary
```
Blob Fields at a Glance

| Field | Type | What it Holds |
|---|---|---|
| `blob.mimetype` | `str` | Data type such as `"text/plain"`, `"application/json"`, `"image/png"` etc. |
| `blob.metadata["uri"]` | `str` | Source URI — where the resource came from |
| `blob.data` | `bytes` | Raw content — works for everything including binary data |
| `blob.as_string()` | `str` | Text content — should only be used for text-based MIME types |

Real Usage Example:
```python
blobs = await client.get_resources("server_name")

for blob in blobs:
    print(f"URI      : {blob.metadata['uri']}")
    print(f"Mimetype : {blob.mimetype}")

    if blob.mimetype == "text/plain":
        print(f"Content  : {blob.as_string()}")

    elif blob.mimetype == "application/json":
        import json
        data = json.loads(blob.as_string())
        print(f"JSON     : {data}")

    elif blob.mimetype.startswith("image/"):
        print(f"Image bytes length: {len(blob.data)}")  # handle as binary

    print("---")
```
---

## 12. Quick Reference

### Agent Creation Cheatsheet
```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-5-nano",              # model string or object
    tools=[tool1, tool2],            # list of callables
    system_prompt="You are...",      # optional
    checkpointer=InMemorySaver(),    # optional, enables memory
    response_format=ToolStrategy(MySchema),  # optional, structured output
    middleware=[...],                # optional, list of middleware
)
```

### Invocation Patterns
```python
# Simple invoke
result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

# With thread memory
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config={"configurable": {"thread_id": "user-123"}}
)

# Streaming
for chunk in agent.stream({"messages": [...]}, stream_mode="updates"):
    print(chunk)

# Async
result = await agent.ainvoke({"messages": [...]})
```

### Common Patterns

```python
# ✅ Use create_agent() for full agentic loops
# ✅ Use bind_tools() when you want manual control
# ✅ Use InMemorySaver() for dev/testing; use persistent DB checkpointer in prod
# ✅ Use stream_mode="updates" to track agent progress step by step
# ✅ Use ProviderStrategy over ToolStrategy when model supports native structured output
# ✅ Use MCP for sharing tools across multiple agents or services
# ⚠️  Always pair ModelCallLimitMiddleware with a checkpointer
# ⚠️  Be careful with response_format + checkpointer — clear state on schema failures
```

---

*Built from LangChain 1.0 documentation. For latest updates, visit [docs.langchain.com](https://docs.langchain.com)*