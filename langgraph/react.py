from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()


@tool
def triple(num: int) -> int:
    """
    This function triples the input number
    Args:
        num: The number to triple
    Returns:
        The triple of the input number
    """

    return float(num * 3)


tools = [triple, TavilySearch(max_results=1)]

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
llm_with_tools = llm.bind_tools(tools)

graph = StateGraph(MessagesState)
