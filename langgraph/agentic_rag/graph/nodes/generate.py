from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()
from graph.state import GraphState

from ..chains.generate import generation_chain


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generates a response to the user's question
    """
    print("--- GENERATE NODE ---")
    question = state["question"]
    documents = state["documents"]
    result = generation_chain.invoke({"question": question, "context": documents})
    return {"generation": result, "question": question, "documents": documents}
