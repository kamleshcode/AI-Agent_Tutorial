# Langchain support multiple schemas types and methods for enforcing structured output.
# 1.TypedDict
# 2.Pydantic
# 3.dataclass

# response format :
# None - No structured output.Agent returns normal messages only.
# ProviderStrategy() - Provider supports schema strongly.No fake tool call generated.
# ToolStrategy() - LangChain converts schema into an internal tool.Agent performs a hidden tool call to fill schema fields.Adds internal
# reasoning/tool trace and slightly increases token usage but improves compatibility across providers.


from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field
import os
from langchain.agents import create_agent

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

class Movie(BaseModel):
    """A movie with details"""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="This year the movie was released")
    director: str = Field(..., description="Director of the movie")
    rating: float = Field(..., description="Rating of the movie out of 10")

# Way 1: Direct Structured Output from Model(Here we directly bind the schema to the LLM)
model_with_structure = model.with_structured_output(Movie,include_raw=True) # include_raw=True : gives the AI message with the parsed message
response = model_with_structure.invoke("Provide details about movie named Inception")
print(response)

# Way 2: Structured Output using Agent(Here agent manages the full execution flow)
agent = create_agent(
    model=model,
    response_format=Movie # LangChain automatically chooses the best structured output strategy internally.
)
response2 = agent.invoke({
    "messages": [
        {"role": "user", "content": "Provide details about movie named Inception"}
    ]
})

print(response2["structured_response"])

# ------------------------Nested Structure------------------------
class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="Budget of movie in INR")

model_with_structure = model.with_structured_output(MovieDetails)
response3 = model_with_structure.invoke("Provide details about movie named KGF")
print(response3)

agent2 = create_agent(
    model=model,
    response_format=MovieDetails
)
response4 = agent2.invoke({
    "messages": [
        {"role": "user", "content": "Provide details about movie named KGF"}
    ]
})
print(response4)