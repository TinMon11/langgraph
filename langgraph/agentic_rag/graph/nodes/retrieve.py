from typing import Any, Dict

from dotenv import load_dotenv

from graph.state import GraphState
from ingestion import retriever


def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print("--- RETRIVE NODE ---")

    question = state["question"]
    # Gets the relevant documents from the ChromaDB embeddings
    documents = retriever.invoke(question)

    # Updates the state with the retrieved documents
    return {"documents": documents, "question": question}
