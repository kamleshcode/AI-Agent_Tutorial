from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

json_schema = {
    "title": "MovieSchema",
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "integer"},
        "director": {"type": "string"},
        "rating": {"type": "number"}
    },
    "required": ["title", "year", "director", "rating"]
}


agent_json = create_agent(
    model=model,
    response_format=json_schema
)

response3 = agent_json.invoke({
    "messages": [
        {"role": "user", "content": "Provide details about Avatar"}
    ]
})

print("\nJSON Schema Output:")
print(response3["structured_response"])

