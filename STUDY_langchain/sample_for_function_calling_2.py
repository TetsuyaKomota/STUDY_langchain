import os

import yaml
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI

with open("tmp/conf.yaml") as f:
    conf = yaml.safe_load(f.read())
    for k, v in conf.items():
        os.environ[k] = v


wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)

wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

tools = [wiki_tool]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": "LangChainについて教えてください"})

print(response["output"])

"""
$ python STUDY_langchain/sample_for_function_calling_2.py

?[1m> Entering new AgentExecutor chain...?[0m
?[32;1m?[1;3m
Invoking: `wikipedia` with `{'query': 'LangChain'}`


?[0m?[36;1m?[1;3mPage: LangChain
Summary: LangChain is a software framework that helps facilitate the integration of ?[0m?[32;1m?[1;3mLangChainは、ソフトウェアフレームワークであり、さまざまなアプリケーショ
ンやシステムにおける統合を支援するためのものです。具体的な詳細については、さらに調査が必要ですが、一般的には異なる技術やサービスを組み合わせて効率的に動作させるためのツール
として利用されます。?[0m

?[1m> Finished chain.?[0m
LangChainは、ソフトウェアフレームワークであり、さまざまなアプリケーションやシステムにおける統合を支援するためのものです。具体的な詳細については、さらに調査が必要ですが、一般
的には異なる技術やサービスを組み合わせて効率的に動作させるためのツールとして利用されます。

"""
