from dataclasses import dataclass
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

@dataclass
class ContactInfo:
    """Contact information of a person"""
    name: str
    email: str
    phone: int

agent = create_agent(
    model=model,
    response_format=ContactInfo,
)

result = agent.invoke({
    "messages":[{"role":"user","content":"Extract contact info from : Kamlesh Patel, abc@gmail.com, (91) 123-456-789"}]
})

print(result["structured_response"])