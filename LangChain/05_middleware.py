"""
Middleware provides a way to more tightly control what happens inside the agent.
Middleware is useful for:
- Tracking agent behavior with logging, analytics and debuging.
- Transforming prompt, tools selection, and output formating.
- Adding retires, fallback, and early termination logic.
- Applying ratelimits, guardrails and PLL detection.
Built-in middleware - Summarization, Human-in-the-loop,Model call limit etc.
"""

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
import os

# # Initialize model
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Summarization Middleware
# - (Automatically summarize history when approaching tokens limits,preserving recent messages while compressing older context)
agent = create_agent(
    model=model,
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 500),
            keep=("messages", 4),
        ),
    ],
)

# create a thread id
config = {"configurable": {"thread_id":"test_1"}}

#Test data
questions = [
    "What is 2+2?",
    "What is 7/2?",
    "What is 3*3?",
    "What is square root of 81?",
    "Multiply 12 and 11",
    "What was my first question?"
]
for q in questions:
    response = agent.invoke(
        {"messages": [HumanMessage(content=q)]},
        config=config
    )
    print(f"\nQUESTION: {q}")
    for msg in response["messages"]:
        print(msg.type.upper(), ":", msg.content)

    print("Total Messages Stored:", len(response["messages"]))

#--------------------------------------------using structured output ---------------------------------------------------
class HotelOutput(BaseModel):
    hotel_name: str
    rating: float = Field(description="Hotel rating out of 5")
    price: float = Field(description="Price per night")

agent2 = create_agent(
    model=model,
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 500),
            keep=("messages", 20),
        )
    ],
    response_format=HotelOutput,
)

config = {"configurable": {"thread_id": "test_1"}}

def count_token(messages):
    total_chars = sum(len(str(m.content)) for m in messages)
    return total_chars // 4

cities = ["Paris", "London"]
for city in cities:
    response2 = agent2.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f"Find one hotel in {city} and return hotel_name, rating and price"
                )
            ]
        },
        config=config,
    )

    tokens = count_token(response2["messages"])
    print(f"\n{city}: {tokens} tokens")

    for msg in response2["messages"]:
        print(msg.type.upper(), ":", msg.content)

    if "structured_response" in response2:
        print("Structured Output:", response2["structured_response"].model_dump())