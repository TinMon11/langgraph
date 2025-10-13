import os
import sys
from typing import List

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

# Add parent directory to path to import react_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from react_agent.callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_string_length(text: str) -> int:
    """
    This tool returns the length of a string

    Args:
        text: The text to get the length of
    """
    text = text.strip("'\n'").strip('"')
    return len(text)


def find_tool_by_name(tools: List[tool], tool_name: str) -> tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


def main():
    print("******** Basic Tool Calling Example ********")

    tools = [get_string_length]

    llm = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0,
        callbacks=[AgentCallbackHandler()],
    )

    messages = [HumanMessage(content="What is the length of the string: DOG")]

    llm_with_tools = llm.bind_tools(tools)
    while True:
        # Get response from LLM
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Check if the response contains tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(" ** Tool calls: ", response.tool_calls)
            # Each tool call contains the function name, params, and id
            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["args"]
                tool_to_use = find_tool_by_name(tools, tool_name)

                # Execute the tool
                observation = tool_to_use.func(**tool_input)

                # Add tool result to messages
                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
                )
        else:
            # No tool calls, we have the final answer
            print(" ** Final Answer: ", response.content)
            break


if __name__ == "__main__":
    main()
