# How the tool calling works under the hood
# When we use the tool calling, we are actually calling the tool function
# We stopped the chat when a 'tool_call' is requested by the agent
# At that point, it has the needed tool name and the inputs
# We can select the tool from the list and call it
# In this code we are doing it manually to understand how it works

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from typing import Union, List
from langchain.agents.format_scratchpad.log import format_log_to_str

load_dotenv()


@tool
def get_string_length(text: str) -> int:
    """
    This tool returns the length of a string
    """
    print(" ** Calling get_string_length with text: ", text)
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

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0,
        stop=["\nObservation:", "Observation:"],
    )

    intermediate_steps = []

    chain = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step: Union[AgentAction, AgentFinish] = chain.invoke(
        {
            "input": "What is the length of the string 'independiente'",
            "agent_scratchpad": intermediate_steps,
        }
    )


    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input
        observation = tool_to_use.func(str(tool_input))

        print(" ** Observation: ", observation)
        intermediate_steps.append((agent_step, observation))
        print(" ** Intermediate steps: ", intermediate_steps)

    agent_step: Union[AgentAction, AgentFinish] = chain.invoke(
        {
            "input": "What is the length of the string 'independiente'",
            "agent_scratchpad": intermediate_steps,
        }
    )

    print(" ** Agent step: ", agent_step)
    
    if isinstance(agent_step, AgentFinish):
        print(" ** Agent finish: ", agent_step.return_values)

if __name__ == "__main__":
    main()
