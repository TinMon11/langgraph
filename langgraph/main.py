from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage

from nodes import run_agent, tool_node
from react import llm_with_tools

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: MessagesState) -> bool:
    """
    This function is used to check if the agent should continue
    """
    return (
        ACT
        if len(state["messages"]) > 1 and state["messages"][LAST].tool_calls
        else END
    )


flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent)
flow.add_node(ACT, tool_node)
flow.add_edge(START, AGENT_REASON)


flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {
        ACT: ACT,
        END: END,
    },
)

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()

with open("graph.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())

if __name__ == "__main__":
    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the temperature in Tokyo in Celsius? List it and triple it. Show me both results, the list and the triple."
                )
            ]
        }
    )
    # print(result["messages"][LAST].content)
    for message in result["messages"]:
        print(message.pretty_print())
