from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

# Chains
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatOpenAI()
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Nodes and Graph
def generation_node(state: MessagesState) -> MessagesState:
    """
    This node is responsible for generating a tweet based on the user's request.
    Args:
        state: The state of the graph
    Returns:
        The state of the graph with the generated tweet
    """
    return {"messages": [generate_chain.invoke(state["messages"])]}


def reflection_node(state: MessagesState) -> MessagesState:
    """
    This node is responsible for reflecting on the generated tweet and providing critique and recommendations.
    Args:
        state: The state of the graph
    Returns:
        The state of the graph with the critique and recommendations
    """
    res = reflect_chain.invoke(state["messages"])
    return {"messages": [HumanMessage(content=res.content)]}


GENERATION = "generation"
REFLECTION = "reflection"


def router(state: MessageGraph) -> str:
    """
    This function is responsible for routing the state of the graph to the appropriate node.
    Args:
        state: The state of the graph
    Returns:
        The name of the node to route to
    """
    return END if len(state["messages"]) > 6 else REFLECTION


graph = StateGraph(MessageGraph)
graph.add_node(GENERATION, generation_node)
graph.add_node(REFLECTION, reflection_node)
graph.add_edge(START, GENERATION)
graph.add_conditional_edges(
    GENERATION,
    router,
    {
        REFLECTION: REFLECTION,
        END: END,
    },
)

graph.add_edge(REFLECTION, GENERATION)


app = graph.compile()

with open("graph.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())

if __name__ == "__main__":

    input = HumanMessage(
        content="Improve the following tweet: 'I'm so excited to be here! presenting the latest Langgraph features!', also add a question to the end of the tweet."
    )

    result = app.invoke({"messages": [input]})
    for message in result["messages"]:
        print(message.pretty_print())
