import json
import os
from textwrap import dedent
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

import yaml
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field, RootModel


class MatchedEntity(BaseModel):
    matched_name: str = Field(..., description="一致したエンティティ名")
    alias_used: str = Field(..., description="テキスト内で使われた表現")
    # context: str = Field(..., description="出現した文章")
    start_index: int = Field(..., description="出現箇所の開始インデックス")
    end_index: int = Field(..., description="出現個所の終了インデックス")


class MatchedEntities(RootModel[list[MatchedEntity]]):
    ...


with open("tmp/conf.yaml") as f:
    conf = yaml.safe_load(f.read())
    for k, v in conf.items():
        os.environ[k] = v

with open("data/entity_db.yaml", encoding="utf-8_sig") as f:
    entity_db = yaml.safe_load(f.read())


def build_chain():
    prompt_path = "STUDY_langchain/prompts/pickup_entities.txt"
    with open(prompt_path, encoding="utf-8_sig") as f:
        prompt = ChatPromptTemplate.from_template(f.read())

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # parser = StrOutputParser()
    parser = PydanticOutputParser(pydantic_object=MatchedEntities)
    format_instructions = parser.get_format_instructions()

    chain = prompt | llm | parser

    return chain, format_instructions


def preprocess_entity_list():
    entity_list = []
    for e in entity_db:
        entity_str_tmpl = "- {name} (別名: {aka}): {desc}"
        entity_list.append(
            entity_str_tmpl.format(
                name=e["name"],
                aka=", ".join(e["alsoknownas"]),
                desc=e["description"],
            )
        )
    entity_list = "\n".join(entity_list)
    return entity_list


def build_vectorstore():
    entity_list = []
    for e in entity_db:
        entity_str_tmpl = "{name} (別名: {aka}): {desc}"
        entity_list.append(
            entity_str_tmpl.format(
                name=e["name"],
                aka=", ".join(e["alsoknownas"]),
                desc=e["description"],
            )
        )
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(entity_list, embeddings)

    return vectorstore


def pickup_candidates(query, vectorstore, k=20):
    docs = vectorstore.similarity_search(query, k)
    entity_list = "\n".join([f"- {d.page_content}" for d in docs])
    return entity_list


def init_model():
    chain, format_instructions = build_chain()
    vectorstore = build_vectorstore()

    def _f(text):
        entity_list = pickup_candidates(text, vectorstore)
        response = chain.invoke(
            {
                "text": text,
                "entity_list": entity_list,
                "format_instructions": format_instructions,
            }
        )
        return ", ".join(sorted([r.matched_name for r in response.root]))

    return _f


if __name__ == "__main__":
    predict = init_model()

    dataset = pd.read_csv("data/news_entity_samples.tsv", sep="\t")

    dataset["predict"] = dataset.body.progress_apply(predict)

    dataset.to_csv("data/predicted.tsv", sep="\t", index=None)
