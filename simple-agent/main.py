from langchain_groq import ChatGroq
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

# Create AI model
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,      # more accurate responses
    groq_api_key=os.environ.get("GROQ_API_KEY"),
)


# Setup MCP server (Firecrawl)
# This configuration tells the client how to start the MCP server
# In this case we run Firecrawl MCP server using NPX
firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
if not firecrawl_key:
    raise ValueError("FIRECRAWL_API_KEY not found in .env file")
server_parameters = StdioServerParameters(
    command="npx",                 # run node tool
    args=["firecrawl-mcp"],        # MCP server name
    env={"FIRECRAWL_API_KEY": firecrawl_key}
)


# Main async function
async def main():
    async with stdio_client(server_parameters) as (read, write):  # Start MCP server
        async with ClientSession(read, write) as session:         # Create MCP session
            await session.initialize()                            # Initialize connection
            tools = await load_mcp_tools(session)                 # Load tools from MCP server(adapter converts MCP tools into LangChain tools.)
            agent = create_react_agent(model, tools)              # Create AI agent
            # System instruction for the agent
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant.You can scrape websites, crawl pages,"
                               "and extract information using Firecrawl tools.Think step-by-step before answering."
                }
            ]
            print("Available Tools:", *[tool.name for tool in tools])
            print("-" * 50)

            # Chat loop
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break
                user_input = user_input[:2000]
                
                # Add user message
                messages.append({
                    "role": "user",
                    "content": user_input
                })
                messages = messages[-10:]

                try:
                    # Run the agent
                    response = await agent.ainvoke({"messages": messages})

                    # Get AI reply
                    ai_message = response["messages"][-1].content

                    # Print reply
                    print("\nAgent:", ai_message)

                    # Save AI reply
                    messages.append({
                        "role": "assistant",
                        "content": ai_message
                    })
                except Exception as e:
                    print("Error:", e)


if __name__ == "__main__":
    asyncio.run(main())