"""TODO"""

import chainlit as cl

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents import AgentType


@cl.on_chat_start
def on_chat_start():
    """TODO"""

    _retriever_func_k = 10

    def retriever_func(query: str) -> str:
        documents = retriever.get_relevant_documents(query, top_k=_retriever_func_k)
        documents = [
            (document.metadata["source"], document.page_content.replace("\n", " "))
            for document in documents
        ]
        return "\n\n".join(
            f"### {source}\n\n{content}" for source, content in documents
        )

    embedding = OpenAIEmbeddings()
    store = FAISS.load_local("store", embeddings=embedding)
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    retriever = MultiQueryRetriever.from_llm(retriever=store.as_retriever(), llm=llm)

    tools = [
        Tool(
            name="Minerva Student Handbook QA System",
            func=retriever_func,
            description="useful for when you need to answer questions about \
                the Minerva student handbook. Input should be a fully formed question.",
        )
    ]
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    cl.user_session.set("chain", agent)


@cl.on_message
async def on_message(message: str):
    """TODO"""

    agent = cl.user_session.get("chain")  # type: AgentExecutor
    response = await agent.acall(
        message, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=response["output"]).send()
