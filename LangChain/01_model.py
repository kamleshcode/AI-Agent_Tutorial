from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from dotenv import load_dotenv
load_dotenv()
import os

#-------------------------------Static Model---------------------------------
# Initialize model
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

print(f'Model Profile : {model.profile}')

response = model.invoke("What is machine learning?")
print(response.content)

# Streaming
for chunk in model.stream("Explain neural network"):
    print(chunk.content, end="")

# Batch
responses = model.batch([
    "What is AI?",
    "What is ML?",
    "What is DL?"
])

for r in responses:
    print(r.content)


#--------------------------------Dynamic Model---------------------------------
small_model = ChatGroq(model="llama-3.1-8b-instant",api_key=os.getenv("GROQ_API_KEY"))
large_model = ChatGroq(model="llama-3.3-70b-versatile",api_key=os.getenv("GROQ_API_KEY"))

@wrap_model_call
def choose_model(request: ModelRequest, handler) -> ModelResponse:
    """Select model based on message count"""
    total_messages = len(request.state["messages"])

    if total_messages > 5:
        selected_model = large_model
        print("Using large model")
    else:
        selected_model = small_model
        print("Using small model")

    return handler(request.override(model=selected_model))

agent = create_agent(
    model=small_model,      # default model
    middleware=[choose_model]
)
response2 = agent.invoke(
    {
        "messages": [
            HumanMessage(content="Hello"),
            HumanMessage(content="What is Groq?"),
            HumanMessage(content="What is DL?"),
            HumanMessage(content="What is Langchain?"),
            HumanMessage(content="What is ML?"),
            HumanMessage(content="Explain me about multi agents?")
        ]
    }
)

for msg in response2["messages"]:
    print(msg.type.upper(), ":", msg.content)