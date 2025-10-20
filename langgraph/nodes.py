from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

load_dotenv()

from react import llm_with_tools, tools

SYSTEM_MESSAGE = """
You're a helpful assistant that can access to tools to answer questions.
"""


def run_agent(state: MessagesState) -> MessagesState:
    """
    This function is used to run the agent
    """

    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_MESSAGE), *state["messages"]]
    )

    return {"messages": [response]}


tool_node = ToolNode(tools)
