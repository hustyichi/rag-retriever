import os
from typing import Any

from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from rag_retriever.modules.doc_store.embedding.localai_embeddings import (
    LocalAIEmbeddings,
)

KB_ROOT_PATH = "./knowledge_base/"


# TODO: 获取配置的模型信息，主要是 api_base_url, api_key
def get_model_info(
    model_name: str = None, platform_name: str = None, multiple: bool = False
) -> dict:
    """
    获取配置的模型信息，主要是 api_base_url, api_key
    如果指定 multiple=True，则返回所有重名模型；否则仅返回第一个
    """
    return {}


# TODO: 获取 api address
def api_address() -> str:
    return "http://127.0.0.1"


def get_Embeddings(
    embed_model: str,
    local_wrap: bool = False,  # use local wrapped api
) -> Embeddings:
    model_info = get_model_info(model_name=embed_model)
    params: dict[str, Any] = dict(model=embed_model)
    if local_wrap:
        params.update(
            openai_api_base=f"{api_address()}/v1",
            openai_api_key="EMPTY",
        )
    else:
        params.update(
            openai_api_base=model_info.get("api_base_url"),
            openai_api_key=model_info.get("api_key"),
            openai_proxy=model_info.get("api_proxy"),
        )

    if model_info.get("platform_type") == "openai":
        return OpenAIEmbeddings(**params)
    elif model_info.get("platform_type") == "ollama":
        return OllamaEmbeddings(
            base_url=model_info.get("api_base_url").replace("/v1", ""),
            model=embed_model,
        )
    else:
        return LocalAIEmbeddings(**params)


def get_kb_path(knowledge_base_name: str):
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)


def get_vs_path(knowledge_base_name: str, vector_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "vector_store", vector_name)


def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")
