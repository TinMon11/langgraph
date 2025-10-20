"""
Reflexion Agent - AI-powered research agent with iterative improvement
This agent generates initial answers, searches for additional information, and revises responses.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import datetime
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_core.output_parsers import JsonOutputToolsParser, PydanticToolsParser
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

load_dotenv()


# =============================================================================
# DATA MODELS
# =============================================================================
class Reflection(BaseModel):
    """Reflection on the quality of an answer."""

    missing: str = Field(
        description="What information is missing to improve the answer?"
    )
    superfluous: str = Field(
        description="What information is superfluous and should be removed?"
    )


class AnswerQuestion(BaseModel):
    """Initial answer with reflection and search queries."""

    answer: str = Field(description="250 word detailed answer to the user's question.")
    reflection: Reflection = Field(description="Reflection on the initial answer.")
    search_queries: list[str] = Field(
        description="1-3 search queries to research information and improve the answer."
    )


class ReviseAnswer(AnswerQuestion):
    """Revised answer with references."""

    references: list[str] = Field(
        description="Citations motivating your updated answer."
    )


# =============================================================================
# TOOLS AND SEARCH
# =============================================================================
tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated search queries using Tavily."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

# =============================================================================
# LLM AND CHAINS
# =============================================================================
llm = ChatOpenAI(model="gpt-4o-mini")
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

# First responder chain
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

# Revisor chain
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

# =============================================================================
# TOOL EXECUTION
# =============================================================================
execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)

# =============================================================================
# GRAPH DEFINITION
# =============================================================================
if __name__ == "__main__":
    MAX_ITERATIONS = 3

    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    # Create the state graph
    graph = StateGraph(AgentState)

    # =============================================================================
    # NODE FUNCTIONS
    # =============================================================================
    def draft_node(state: AgentState) -> AgentState:
        """Generate initial answer with tool calls."""
        result = first_responder.invoke(state)
        return {"messages": state["messages"] + [result]}

    def revise_node(state: AgentState) -> AgentState:
        """Revise answer based on search results."""
        result = revisor.invoke(state)
        return {"messages": state["messages"] + [result]}

    # Add nodes to graph
    graph.add_node("draft", draft_node)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("revise", revise_node)

    # =============================================================================
    # ROUTING FUNCTIONS
    # =============================================================================
    def should_continue(state: AgentState) -> str:
        """Decide whether to execute tools or end after draft."""
        messages = state["messages"]
        last_message = messages[-1]
        # If the LLM makes a tool call, then we route to the "execute_tools" node
        if (
            hasattr(last_message, "tool_calls")
            and last_message.tool_calls
            and len(last_message.tool_calls) > 0
        ):
            return "execute_tools"
        # Otherwise, we stop
        return END

    def event_loop(state: AgentState) -> str:
        """Control the iteration loop for revision cycles."""
        count_tool_visits = sum(
            isinstance(item, ToolMessage) for item in state["messages"]
        )
        if count_tool_visits >= MAX_ITERATIONS:
            return END
        return "execute_tools"

    # =============================================================================
    # GRAPH EDGES
    # =============================================================================
    # Initial flow
    graph.add_edge(START, "draft")

    # Conditional routing from draft
    graph.add_conditional_edges(
        "draft",
        should_continue,
        {
            "execute_tools": "execute_tools",
            END: END,
        },
    )

    # Flow from tools to revision
    graph.add_edge("execute_tools", "revise")

    # Conditional routing from revision (iteration control)
    graph.add_conditional_edges(
        "revise",
        event_loop,
        {
            "execute_tools": "execute_tools",
            END: END,
        },
    )

    # =============================================================================
    # EXECUTION
    # =============================================================================
    app = graph.compile()

    # Generate graph visualization
    with open("graph.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())

    # Run the agent
    human_message = HumanMessage(content="Write about AI-Powered Logistic Solutions")

    result = app.invoke({"messages": [human_message]})

    print("=" * 50)
    print("FINAL RESULT")
    print("=" * 50)
    for message in result["messages"]:
        print(message.pretty_print())
