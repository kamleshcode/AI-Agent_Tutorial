import os
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment")

if not os.getenv("FIRECRAWL_API_KEY"):
    raise ValueError("FIRECRAWL_API_KEY not found in environment")


# Create reasoning engine using Groq language model.
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Configure MCP client to connect with Firecrawl using local stdio transport.
client = MultiServerMCPClient(
    {
        "firecrawl": {
            "transport": "stdio",
            "command": "npx",
            "args": ["firecrawl-mcp"],
            "env": {
                "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")
            }
        }
    }
)

# Store conversation memory in RAM.
memory = InMemorySaver()

async def main():
    """
    Run async chat loop with MCP tools and in-memory conversation tracking.
    """
    tools = await client.get_tools()

    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=memory,
        system_prompt="""
            You are a helpful AI assistant.

            You have access to Firecrawl MCP tools for:
            - scraping webpages
            - crawling websites
            - extracting structured data

            Rules:
            1. Use Firecrawl only when website data is required.
            2. Answer directly for general knowledge.
            3. Keep responses concise.
        """
    )

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        try:
            response = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input[:1500]
                        }
                    ]
                },
                config={
                    "configurable": {
                        "thread_id": "chat1"
                    }
                }
            )

            print("\nAgent:", response["messages"][-1].content)

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    asyncio.run(main())