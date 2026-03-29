from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
import os

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# to create tool we use @tool decorator above the function
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# We can connect llm and tools in 2 ways
# 1.bind_tools()
# 2.create_agent()

# METHOD 1: bind_tools()
print("\n=== METHOD 1 : bind_tools() ===")
model_with_tools = model.bind_tools([multiply])

response = model_with_tools.invoke("Multiply 4 and 5")

print(response)
print("Tool Calls:", response.tool_calls)

# METHOD 2: create_agent()
print("\n=== METHOD 2 : create_agent() ===")

agent = create_agent(
    model=model,
    tools=[multiply],
    system_prompt="You are a math assistant"
)

agent_response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Multiply 8 and 6"}
    ]
})

for msg in agent_response["messages"]:
    print(msg.type.upper(), ":", msg.content)