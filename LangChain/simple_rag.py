from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("example.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding=embeddings)

retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using the context below.
    
    Context:
    {context}
    
    Question:
    {question}
    """
)

llm = ChatOpenAI(model="gpt-4o-mini")

rag_chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is the main topic of example.txt?")
print(response)