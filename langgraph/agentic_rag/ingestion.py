from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

docs_splits = text_splitter.split_documents(docs_list)

# I run it only once to seed the Chroma db
# Commented out to avoid doing it every time we call the functions as its already there
vectorstore = Chroma.from_documents(
    documents=docs_splits,
    embedding=OpenAIEmbeddings(),
    collection_name="rag-chroma",
    persist_directory="./.chroma",
)

retriever = Chroma(
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
    collection_name="rag-chroma",
).as_retriever()


if __name__ == "__main__":
    print("Ingesting...")
