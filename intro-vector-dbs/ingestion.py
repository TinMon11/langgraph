import os

from dotenv import load_dotenv
# Text Loader (util for opening a file, in this case, text)
from langchain_community.document_loaders import TextLoader
# Embedding Model (we need to convert the text into embeddings to store them on the vector db)
from langchain_community.embeddings import OpenAIEmbeddings
# Pinecode will be our vector db
from langchain_pinecone import PineconeVectorStore
# Text Splitter (we need to split the text into chunks, to avoid token limits)
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    # 1. We first need to create the loader
    loader = TextLoader(
        r"C:\Users\HP\Documents\PersonalGithub\langgraph-course\intro-vector-dbs\mediumblog1.txt",
        encoding="utf-8",
    )
    # 2. Load the documents as a Langchain document
    documents = loader.load()

    # 3. We need to split the text into chunks. So we create the splitter
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,  # Each chunk will have 1000 tokens
        chunk_overlap=200,  # We will overlap the chunks by 200 tokens
    )

    # 4. We now split the text into chunks
    text_chunks = text_splitter.split_documents(documents)

    # 5. We now need to create the embedding model
    # This will use under the hood the open ai model to create the embeddings
    embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Store them on Pinecone (our vector db)
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embedding_model,
        index_name=os.getenv("INDEX_NAME"),
    )

    print("Documents ingested successfully")
