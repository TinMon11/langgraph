from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langchain import hub
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor

load_dotenv()


def main():
    print("Running Create React Agent")

    llm_model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.25)

    react_prompt = hub.pull("hwchase17/react")

    search_tool = TavilySearch(max_results=2)

    # llm_model_with_tools = llm_model.bind_tools([search_tool])

    react_agent = create_react_agent(llm=llm_model, tools=[search_tool], prompt=react_prompt)

    agent_executor = AgentExecutor(agent=react_agent, tools=[search_tool], verbose=True)

    result = agent_executor.invoke(
        {"input": "How much money did US swapped with Argentina yesterday?"}
    )

    print(result)


if __name__ == "__main__":
    main()
