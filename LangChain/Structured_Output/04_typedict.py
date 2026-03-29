from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os
from typing_extensions import TypedDict, Annotated

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

class MovieDict(TypedDict):
    title: Annotated[str, ..., "The title of the movie"]
    year: Annotated[int, ..., "The release year of the movie"]
    director: Annotated[str, ..., "Director of the movie"]
    rating: Annotated[str, ..., "Rating of the movie"]

# Direct Structured Output from Model
model_with_typedict = model.with_structured_output(MovieDict)
response = model_with_typedict.invoke("Please provide me the details of movie Avengers")
print(response)

# Structured Output using Agent
agent = create_agent(
    model=model,
    response_format=MovieDict
)
response2 = agent.invoke({
    "messages": [
        {"role": "user", "content": "Provide details about movie Avengers"}
    ]
})

print(response2["structured_response"])

#-------------------------Nested Structure-------------------------
class Actor(TypedDict):
    name: str
    role: str

class MovieDetails(TypedDict):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: Annotated[float, ..., "The budget of the movie"]

agent2 = create_agent(
    model=model,
    response_format=MovieDetails
)

response2 = agent2.invoke({
    "messages": [
        {"role": "user", "content": "Provide details about movie named KGF"}
    ]
})