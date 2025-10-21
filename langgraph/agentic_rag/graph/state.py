from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of the graph.
    Attributes:
        - question: The question to answer
        - generation: LLM generation
        - web_search: wether to add search
        - documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
