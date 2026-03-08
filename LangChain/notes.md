# LangChain
- LangChain is a Python framework that helps you build applications using LLMs like GPT-4.
- It connects models with tools like APIs, databases, and other systems to perform complete tasks.
- LangChain makes working with LLMs easier by breaking tasks into smaller, manageable parts called prompts, chains, and agents.
- Recommended Python Version:
  - Python 3.10 – 3.12 (best compatibility with the LangChain ecosystem)

**Prompts**: Templates that tell the LLM exactly what to do (like giving instructions to a helper).  
**Chains**: Sequences of steps where each step may involve using the LLM, calling an API, or doing a calculation.  
**Agents**: Smart decision-makers that figure out what to do next, like deciding which tool or step to use.

```bash
                    Vector Stores
    Document Loader        |        Prompts
                  \       |       /
                   \      |      /
                      LangChain
                    /     |     \
                  /       |      \
              LLMs        |       Agents
                        Chains
```

- LangChain is like a wrapper on top of openai API.
Example:
```python
from langchain_openai import OpenAI
# Create LLM client
llm = OpenAI()
# Zero-shot prompt
prompt = "Can you tell me the total number of countries in Asia?"
print(llm.predict(prompt).strip())
```
- In LangChain, you can get output from an LLM in two ways:
  1. ```predict``` – Use for simple, single-string prompts.
  ```python
  response = llm.predict("What is the capital of France?")
  print(response)
  ```
  2. ```invoke``` – Use when working with templates, chains, or structured workflows.
  ```python
  response = llm.invoke({"country": "France"})
  print(response.content)
  ```

## Prompt templates(2ways)
- Prompt Template is like a fill in the blank guide used in generativeAI.
- It provides the clear structure or set of instruction to help the AI understand what kind of response to create.
- These templates can be reused and customized by changing certain parts(like fill in the blank) to get result while keeping the overall format same.
Way 1: Using PromptTemplate object
```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")
# Create prompt template
prompt_template = PromptTemplate(
    input_variables=["country"],
    template="Can you tell me the capital of {country}?"
)
# Format the template
prompt = prompt_template.format(country="India")
# Call the LLM
response = llm.invoke(prompt)
print(response.content)
```
Way 2: Using from_template shortcut
```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
# Create prompt template using shortcut
prompt = PromptTemplate.from_template(
    "What is a good name for a company that makes {product}?"
)
formatted_prompt = prompt.format(product="toys")
response = llm.invoke(formatted_prompt)
print(response.content)
```
## Agents
- is a core concept that refers to AI system capable of dynamically deciding how to interact with tools,apis,or other components to accomplish the task.
- Key Features:
1. Dynamic Decision making
2. Tool usage
3. Reasoning and execution
4. Integration
- so therefore, here we use agent concept it will interact with 3rd party api and we will get realtime o/p.
- Agents work using a Reasoning Pattern called: ReAct (Reason + Act)
- Example tools:
  - Wikipedia
  - Google Search
  - Python execution
  - SQL database
  - APIs
  - RAG retrievers

Without tools, agents cannot access real-time information.
- Example:
```python
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
# create LLM
llm = ChatOpenAI(model="gpt-4o-mini")
# create tool
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [wiki]

# create agent
agent = create_react_agent(llm, tools)

# executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

agent_executor.invoke({"input": "What is the GDP of the USA?"})
```

## Chain
- A chain in LangChain is a series of connected components or steps that work together to complete a task.
- Each step takes input, processes it, and passes the result to the next step
- example:
```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = PromptTemplate.from_template(
    "Suggest a name for a company that makes {product}"
)
chain = prompt | llm
response = chain.invoke({"product": "wine"})
print(response.content)
```
### Sequential Chain
- If we want to combine multiple chains in sequence, we can compose them using LCEL instead of the deprecated SimpleSequentialChain.
```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Step 1: Define prompts
name_prompt = PromptTemplate.from_template(
    "Suggest a startup name for {idea}"
)

strategy_prompt = PromptTemplate.from_template(
    "Suggest business strategies for a company called {name}"
)

# Method 1: Separate chains
name_chain = name_prompt | llm
strategy_chain = strategy_prompt | llm

# Step-by-step execution
startup_name = name_chain.invoke({"idea": "Artificial Intelligence"}).content
result = strategy_chain.invoke({"name": startup_name})

print("Step-by-step chain output:")
print(result.content)

# Method 2: LCEL composition
# Combine both chains into a single pipeline
combined_chain = name_prompt | llm | strategy_prompt | llm

# Single invocation with LCEL
lc_el_result = combined_chain.invoke({"idea": "Artificial Intelligence"})

print("\nLCEL composition output:")
print(lc_el_result.content)
```
1. Step-by-step chains:
   - Execute each chain individually.
   - Good if you want to inspect outputs between steps or apply conditional logic.

2. LCEL composition:
   - Combine multiple steps into a single chain using |.
   - More concise and clean for sequential pipelines.
   - Input flows automatically from one step to the next.
   
### Conversation Chain
- Conversation chains maintain dialogue context.
```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory
)

conversation.invoke({"input": "Who won the first cricket world cup?"})
conversation.invoke({"input": "Who was the captain of the winning team?"})
```

## Document Loader
- Document Loader in langchain are tools that help you load and process data from different sources(like text files, PDFs,website or DB)
  so it can be used with a language model.
- Common Types:
  - Files : Load data from text,CSV or PDF files.
  - Web :Load content from website or sitemaps
  - Database :Fetch data from SQL or MongoDB
  - Cloud : Load files from cloud stroges(S3,Azure, GoogleCloud)
- Modern LangChain Loaders:
  - CSVLoader → for CSV
  - PDFLoader / PyPDFLoader → for PDF
  - UnstructuredURLLoader → for websites
  - S3FileLoader → for cloud storage

```python
from langchain.document_loaders import TextLoader
# Load a text file
loader = TextLoader("example.txt")
documents = loader.load()
# Print the content of the first document
print(documents[0].page_content)
```

## Memory
- Memory in LangChain refers to the ability to store and recall information from previous interactions, enabling conversational AI to maintain context.
across multiple messages in a session.
- Types of Memory in LangChian:
  - **ConversationBufferMemory** : stores the entire conversion history as a single text buffer.
  - **ConversationBufferWindowMemory** : keeps only the last few interactions (sliding window).
  - **ConversationSummaryMemory** : summarize previous interaction into concise format.
  - **ConversationKnowledgeGraphMemory** : build a knowledge graph from the conversion to track relationship between entities.
  - **CombinedMemory** : combines multiple types of memory for more complex scenarios.
- Example:
```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Create memory
memory = ConversationBufferMemory()

# Conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

# Example prompt template
prompt_template = PromptTemplate(
    input_variables=["product"],
    template="What would be a good name for a company that makes {product}?"
)

# Run the chain with different products
response1 = conversation.invoke({"input": prompt_template.format(product="Wine")})
response2 = conversation.invoke({"input": prompt_template.format(product="Camera")})

# Print conversation memory
print(memory.buffer)
```
output:
```bash
Human: What would be a good name for a company that makes Wine?
AI: Vineyard Cellar
Human: What would be a good name for a company that makes Camera?
AI: Camera Lumen Technologies
```
## Vector Stores & Embeddings
- Converts text into numbers (vectors) so LLM can search and compare information easily.
- Helps LLMs answer questions using real documents (RAG).
- Example:
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

texts = ["This is document 1", "This is document 2"]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(texts, embedding=embeddings)

retriever = vectorstore.as_retriever(k=1)
docs = retriever.get_relevant_documents("Tell me about document 1")
print(docs[0].page_content)
```

## Callbacks & Logging
- Lets you track what the LLM is doing step by step.
- Example:
```python
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain

callback = StdOutCallbackHandler()
chain = LLMChain(llm=llm, prompt=prompt_template, callbacks=[callback])
chain.run({"product": "Camera"})
```
# RAG Pipeline with LangChain
- Combines a document retriever + LLM to answer questions from real documents.
- Components of a RAG System
  1. Document Loaders
  - Load your source data (PDFs, CSVs, websites, DBs, cloud).
  - Example: ```TextLoader```, ```CSVLoader```, ```UnstructuredURLLoader```.

  2. Vector Stores & Embeddings
  - Convert your documents into vectors so the LLM can “search” them.
  - Example: ```Chroma```, ```FAISS```,```Pinecone``` with ```OpenAIEmbeddings```.

  3. Retriever
  - Retrieves the most relevant document chunks based on a query.
  - Acts like a “smart search” before the LLM generates answers.

  4. Prompt / Chains
  - Chains combine the retriever + LLM with a prompt template.
  - Guides the LLM to answer based on retrieved documents.

  5. Memory
  - Store context from previous questions if you want multi-turn RAG (like chat history).

  6. Agents / Tools (optional)
  - Extend RAG with tools like calculators, Wikipedia search, or APIs for real-time info.

- Example:
```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain_openai import ChatOpenAI

# 1. Load documents
loader = TextLoader("example.txt")
documents = loader.load()

# 2. Embed and store in vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embedding=embeddings)

# 3. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 4. Create LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# 5. Combine into RAG chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 6. Ask a question
result = qa_chain.run("What is the main topic of example.txt?")
print(result)
```
## Version Compatibility
| LangChain Version | Code Compatibility | Key Features / Notes |
|------------------|-----------------|--------------------|
| <1.0             | Older syntax    | Only `.predict()` available, no `.invoke()`, limited agents and chains, memory features basic |
| 1.2.10           | Current code    | Uses `langchain_openai` & `langchain`, supports `.predict()` & `.invoke()`, LLMChain + Memory + Callbacks, RAG pipelines, ReAct agents |
| 2.x+ (future)    | Upcoming        | Possible module/API changes, async improvements in chains & vectorstores, LangGraph integration, updated agent tooling |

---
Code in this tutorial is compatible with **LangChain v1.2.10**
py -3.12 -m pip install virtualenv
