from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("--- GRADE DOCUMENTS NODE ---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False

    for d in documents:
        result = retrieval_grader.invoke({"documents": d, "question": question})
        if result.binary_score == "yes":
            filtered_docs.append(d)
        else:
            web_search = True
            continue

    return {"documents": filtered_docs, "web_search": web_search, "question": question}
