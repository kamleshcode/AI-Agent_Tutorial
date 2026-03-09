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
    Document Loader       |        Prompts
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
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = "Can you tell me the total number of countries in Asia?"
response = llm.invoke(prompt)
print(response.content)
```

## Prompt templates(2ways)
- Prompt Template is like a fill in the blank guide used in generativeAI.
- It provides the clear structure or set of instruction to help the AI understand what kind of response to create.
- These templates can be reused and customized by changing certain parts(like fill in the blank) to get result while keeping the overall format same.
Way 1: Using PromptTemplate object
```python
from langchain_core.prompts import PromptTemplate
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
from langchain_core.prompts import PromptTemplate
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
1. Dynamic Decision-making
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
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
llm = ChatOpenAI(model="gpt-4o-mini")
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
agent = create_agent(
    model=llm,
    tools=[wiki],
)
response = agent.invoke({"messages": [{"role": "user", "content": "What is the GDP of the USA?"}]})
print(response)
```

## Chain
- A chain in LangChain is a series of connected components or steps that work together to complete a task.
- Each step takes input, processes it, and passes the result to the next step
- example:
```python
from langchain_core.prompts import PromptTemplate
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
from langchain_core.prompts import PromptTemplate
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
from langchain_community.document_loaders import TextLoader
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
- Conversation memory now works with RunnableWithMessageHistory.
- Example:
```python
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

chain = prompt | llm

history = InMemoryChatMessageHistory()

chat = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input"
)

chat.invoke(
    {"input": "Who won the first cricket world cup?"},
    config={"configurable": {"session_id": "1"}}
)

chat.invoke(
    {"input": "Who was the captain?"},
    config={"configurable": {"session_id": "1"}}
)
```

## Vector Stores & Embeddings
- Converts text into numbers (vectors) so LLM can search and compare information easily.
- Helps LLMs answer questions using real documents (RAG).
- Example:
```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

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
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# callback handler
callback = StdOutCallbackHandler()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    callbacks=[callback]
)
prompt = ChatPromptTemplate.from_template(
    "Suggest a good name for a company that makes {product}"
)
# Runnable pipeline
chain = prompt | llm
response = chain.invoke({"product": "Camera"})
print(response.content)
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
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load documents
loader = TextLoader("example.txt")
documents = loader.load()
# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.split_documents(documents)
# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding=embeddings)
# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
# Prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the question using the context below.
Context:
{context}
Question:
{question}
"""
)
# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini")
# Build RAG pipeline
rag_chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)
# Ask a question
response = rag_chain.invoke("What is the main topic of example.txt?")
print(response)
```
# Modern LangChain Architecture
Old LangChain
-------------
Prompt → LLMChain → Output

Modern LangChain
----------------
Prompt | Model | Parser

---
Code in this tutorial is compatible with **LangChain v1.2.10**
- ```py -3.12 -m pip install virtualenv```
