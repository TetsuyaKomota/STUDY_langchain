import os
from collections import Counter

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

import yaml
from langchain.output_parsers import PydanticOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field, RootModel


class Theme(BaseModel):
    name: str = Field(..., description="テーマ名")
    relevance: float = Field(..., ge=0.0, le=1.0, description="主題度(0~1)")


class ThemePrediction(BaseModel):
    predicted_themes: list[Theme] = Field(..., description="推測されたテーマ")
    reason: str = Field(..., description="判断の根拠となる説明")


with open("tmp/conf.yaml") as f:
    conf = yaml.safe_load(f.read())
    for k, v in conf.items():
        os.environ[k] = v

with open("data/topic_db.yaml", encoding="utf-8_sig") as f:
    topic_db = yaml.safe_load(f.read())


def build_vectorstore():
    embeddings = OpenAIEmbeddings()

    if os.path.exists("tmp/topic_vs_index"):
        vectorstore = FAISS.load_local(
            "tmp/topic_vs_index", embeddings, allow_dangerous_deserialization=True
        )
        return vectorstore

    documents = []
    for t in topic_db:
        meta = {"theme": t["theme"]}
        doc = Document(page_content=t["article"], metadata=meta)
        documents.append(doc)

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("tmp/topic_vs_index")

    return vectorstore


def pickup_candidates(query, vectorstore, k=20):
    docs = vectorstore.similarity_search(query, k)
    candidates = Counter(d.metadata["theme"] for d in docs)
    return dict(candidates)


def build_chain():
    prompt_path = "STUDY_langchain/prompts/pickup_theme_candidates.txt"
    with open(prompt_path, encoding="utf-8_sig") as f:
        prompt = ChatPromptTemplate.from_template(f.read())

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    parser = PydanticOutputParser(pydantic_object=ThemePrediction)
    format_instructions = parser.get_format_instructions()

    chain = prompt | llm | parser

    return chain, format_instructions


def init_model():
    chain, format_instructions = build_chain()
    vectorstore = build_vectorstore()

    def _f(text):
        theme_counts = pickup_candidates(text, vectorstore)
        response = chain.invoke(
            {
                "text": text,
                "theme_counts": theme_counts,
                "format_instructions": format_instructions,
            }
        )
        return response

    return _f


if __name__ == "__main__":
    predict = init_model()

    dataset = pd.read_csv("data/topic_samples.tsv", sep="\t")

    dataset["tmp"] = dataset.body.progress_apply(predict)
    dataset["predicted"] = dataset.tmp.apply(
        lambda x: ",".join(
            sorted(set([t.name for t in x.predicted_themes if t.relevance >= 0.5]))
        )
    )
    dataset["reason"] = dataset.tmp.apply(lambda x: x.reason)
    dataset = dataset.drop(columns="tmp")

    dataset.to_csv("data/predicted_topic.tsv", sep="\t", index=None)
