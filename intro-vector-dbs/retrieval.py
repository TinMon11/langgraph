from dotenv import load_dotenv
# LangChain hub (for pulling pre-built prompts and chains)
from langchain import hub
# Chain for combining multiple documents into a single prompt
from langchain.chains.combine_documents import create_stuff_documents_chain
# Chain for retrieving relevant documents and then processing them
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
# RunnablePassthrough is a runnable that passes through the input as is
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


load_dotenv()
import os

if __name__ == "__main__":
    print("Retrieving...")

    # 1. Create the LLM model for generating responses
    llm_model = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-nano"
    )

    query = "What is Pincone in machine learning?"

    # 2. Create embeddings model (same as in ingestion.py)
    # This will convert text to vectors for similarity search
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # 3. Connect to our Pinecone vector store (where we stored the documents)
    vector_store = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"), embedding=embeddings
    )

    # 4. Create a retriever from the vector store
    # This converts the vector store into a retriever that can be used with chains
    retriever = vector_store.as_retriever()

    # 5. Pull a pre-built prompt template from LangChain hub
    # This prompt is specifically designed for retrieval-augmented generation (RAG)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # 6. Create a chain that takes a list of documents and formats them into a prompt
    # This chain will "stuff" the retrieved documents into the prompt template
    combine_docs_chain = create_stuff_documents_chain(
        llm_model, retrieval_qa_chat_prompt
    )

    # 7. Create the final retrieval chain that:
    # - Takes a query
    # - Retrieves relevant documents from vector store
    # - Combines them with the query using the prompt template
    # - Sends everything to the LLM for a contextual response
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # 8. Now we can use the retrieval chain to get contextual answers
    # This will search for relevant documents and use them to answer the query
    print("\n--- Using RAG (Retrieval Augmented Generation) ---")
    rag_result = retrieval_chain.invoke({"input": query})
    print("AI (with retrieval):", rag_result["answer"])

    # ---- Using a custom chain LCEL

    template = """
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer as concise as possible.
    Always say 'thanks for asking!' at the end of the answer.
    
    {context}
    
    Question: {question}
    
    Useful Answer:
    """

    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": vector_store.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | custom_rag_prompt
        | llm_model
    )

    result = rag_chain.invoke(query)

    print("AI (with retrieval):", result.content)
