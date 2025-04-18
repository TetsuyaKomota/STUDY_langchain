import asyncio
import os

import yaml
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

with open("tmp/conf.yaml") as f:
    conf = yaml.safe_load(f.read())
    for k, v in conf.items():
        os.environ[k] = v


server_params = StdioServerParameters(
    command="python", args=["STUDY_langchain/math_server.py"]
)


async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)

            llm = ChatOpenAI(model="gpt-4o", temperature=0)

            prompt = hub.pull("hwchase17/openai-functions-agent")

            agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            response = await agent_executor.ainvoke({"input": "(3 + 5) x 12 はいくつですか？"})

            print(response["output"])


asyncio.run(run_agent())

"""
$ python STUDY_langchain/sample_for_function_calling_3.py
E:\STUDY_langchain\tmp\venv_study_langchain\Lib\site-packages\langsmith\client.py:280:
 LangSmithMissingAPIKeyWarning: API key must be provided when using hosted LangSmith A
PI
  warnings.warn(


?[1m> Entering new AgentExecutor chain...?[0m
?[32;1m?[1;3m
Invoking: `add` with `{'a': 3, 'b': 5}`


?[0m?[36;1m?[1;3m8?[0m?[32;1m?[1;3m
Invoking: `multiply` with `{'a': 8, 'b': 12}`


?[0m?[33;1m?[1;3m96?[0m?[32;1m?[1;3m(3 + 5) × 12 は 96 です。?[0m

?[1m> Finished chain.?[0m
(3 + 5) × 12 は 96 です。
"""
