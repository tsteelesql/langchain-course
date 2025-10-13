from dotenv import load_dotenv
load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


tools = [TavilySearch()]
llm = ChatOllama(model='llama3.2')
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor

# def main():
#     print("Hello from langchain-course!")

def main():
    result = chain.invoke(
        input={
            "input":"search for 3 job postings for an ai engineer using langchain in Columbus Ohio on Linkedin and list their details",
        }
    )


if __name__ == "__main__":
    main()
