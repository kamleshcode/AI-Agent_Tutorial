from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StdOutCallbackHandler
from langchain.agents import create_react_agent, AgentExecutor
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

def main():
    # 1. Load documents
    loader = TextLoader("example.txt")
    documents = loader.load()

    # 2. Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 3. Initialize LLM and logging
    llm = ChatOpenAI(model="gpt-4o-mini")
    callback = StdOutCallbackHandler()

    # 4. Initialize memory
    memory = ConversationBufferMemory()

    # 5. Prompt template for LLMChain
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Answer the question based on context below:\n\nContext:\n{context}\n\nQuestion: {question}"
    )

    # 6. LLMChain for refining answers
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory,
        callbacks=[callback]
    )

    # 7. RetrievalQA chain (RAG)
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    # 8. Wikipedia Agent
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = [wiki_tool]
    agent = create_react_agent(llm, tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 9. Interactive loop
    print("Mini RAG Agent Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("\nAsk a question: ")
        if user_input.lower() == "exit":
            break

        # RAG chain for document-based answer
        rag_answer = rag_chain.run(user_input)

        # Refine RAG answer using memory-enabled LLMChain
        refined_answer = chain.run({
            "context": rag_answer,
            "question": user_input
        })

        # Agent fallback for real-time knowledge
        agent_answer = agent_executor.invoke({"input": user_input}).content

        # 10. Print all outputs
        print(f"\n[RAG Answer]: {rag_answer}")
        print(f"[Refined Answer]: {refined_answer}")
        print(f"[Agent Answer]: {agent_answer}")

if __name__ == "__main__":
    main()