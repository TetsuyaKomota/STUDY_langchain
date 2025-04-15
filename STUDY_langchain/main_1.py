import json
import os
from textwrap import dedent

import yaml
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
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

with open("tmp/entity_db.yaml", encoding="utf-8_sig") as f:
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


if __name__ == "__main__":
    chain, format_instructions = build_chain()
    entity_list = preprocess_entity_list()

    text = "トヨタは新型の電気自動車を2025年に発表予定である．岸田ちゃんも喜んでいる"
    response = chain.invoke(
        {
            "text": text,
            "entity_list": entity_list,
            "format_instructions": format_instructions,
        }
    )

    print(response)
