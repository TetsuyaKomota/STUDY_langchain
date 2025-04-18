import os

import yaml
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

with open("tmp/conf.yaml") as f:
    conf = yaml.safe_load(f.read())
    for k, v in conf.items():
        os.environ[k] = v


@tool
def get_weather(city: str) -> str:
    """
    指定された都市の現在の天気を返します
    """
    return f"{city} の天気は 晴れ ，気温は 25℃ です"


tools = [get_weather]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": "東京の天気を教えてください"})

print(response["output"])

"""
$ python STUDY_langchain/sample_for_function_calling.py

?[1m> Entering new AgentExecutor chain...?[0m
?[32;1m?[1;3m
Invoking: `get_weather` with `{'city': '東京'}`


?[0m?[36;1m?[1;3m東京 の天気は 晴れ ，気温は 25℃ です?[0m?[32;1m?[1;3m東京の現在の天気は晴れで、気温は25℃です。?[0m

?[1m> Finished chain.?[0m
東京の現在の天気は晴れで、気温は25℃です。

"""
