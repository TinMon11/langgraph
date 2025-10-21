from typing import Any, Dict

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch

load_dotenv()
from graph.state import GraphState

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Executes a web search if the documents are not relevant to the question
    """
    print("--- WEBSEARCH NODE ---")
    question = state["question"]
    documents = state["documents"]
    web_search = state["web_search"]
    if web_search:
        results = web_search_tool.invoke(question)
        # Handle different result formats from TavilySearch
        if isinstance(results, list):
            # If results is a list of documents
            joined_results = "\n".join(
                [
                    r.page_content if hasattr(r, "page_content") else str(r)
                    for r in results
                ]
            )
        else:
            # If results is a single document or string
            joined_results = (
                results.page_content
                if hasattr(results, "page_content")
                else str(results)
            )

        web_results = Document(page_content=joined_results)

        if documents:
            documents.append(web_results)
        else:
            documents = [web_results]

    return {"documents": documents, "question": question}
