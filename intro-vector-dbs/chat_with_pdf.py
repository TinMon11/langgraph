import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader

# Helps to convert pdf files to perform querys faster
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Chatting with PDF...")

    pdf_path = r"C:\Users\HP\Documents\PersonalGithub\langgraph-course\intro-vector-dbs\react.pdf"

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separator="\n")

    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # This will store it in the RAM of the machine
    vector_store = FAISS.from_documents(docs, embeddings)

    # We can also store it
    vector_store.save_local("faiss_index_react")

    # Load from local (just to showcase how it works)
    vector_store = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    llm = ChatOpenAI(model="gpt-4.1-nano")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(), combine_docs_chain
    )

    result = retrieval_chain.invoke({"input": "Give me the gist of React in 3 sentences"})
    print(result["answer"])
