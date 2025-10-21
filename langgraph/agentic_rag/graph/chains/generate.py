from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from graph.state import GraphState

llm = ChatOpenAI(model="gpt-4.1-nano")

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generates a response to the user's question
    """
    print("--- GENERATE NODE ---")
    question = state["question"]
    documents = state["documents"]
    result = generation_chain.invoke({"question": question, "context": documents})
    return {"generation": result, "question": question, "documents": documents}
