from langchain_groq import ChatGroq
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment")

# Create AI model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,      # more accurate responses
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

async def main():
    async with stdio_client(server_parameters) as (read, write):  # Start MCP server
        async with ClientSession(read, write) as session:         # Create MCP session
            await session.initialize()                            # Initialize connection
            tools = await load_mcp_tools(session)                 # Load tools from MCP server(adapter converts MCP tools into LangChain tools.)
            agent = create_agent(
                model=model,
                tools=tools,
                system_prompt="""
            You are a helpful AI assistant.

            You have access to Firecrawl tools that allow you to:
            - scrape webpages
            - crawl websites
            - extract structured information.
            
            Guidelines:
            1. Use Firecrawl tools ONLY when the user asks for information from a specific website or webpage.
            2. For general knowledge questions, answer directly without using tools.
            3. Think step-by-step before deciding whether a tool is needed.
            4. Provide clear and concise answers to the user.
            """
            )

            messages = []

            #chat loop
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break
                user_input = user_input[:1500]

                # Add user message
                messages.append({
                    "role": "user",
                    "content": user_input
                })

                try:
                    response = await agent.ainvoke({"messages": messages}) # Run the agent
                    ai_message = response["messages"][-1].content          # Get AI reply
                    print("\nAgent:", ai_message)
                    # Save AI reply
                    messages.append({
                        "role": "assistant",
                        "content": ai_message
                    })
                    messages = messages[-6:]
                except Exception as e:
                    print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())