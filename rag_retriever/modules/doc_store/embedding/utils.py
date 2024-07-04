import os
from pathlib import Path
from typing import Any, List

from langchain.schema import Document
from langchain.text_splitter import TextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from rag_retriever.modules.doc_store.embedding.localai_embeddings import (
    LocalAIEmbeddings,
)
from rag_retriever.modules.loaders.loaders import (
    SUPPORTED_EXTS,
    get_loader,
    get_loader_by_name,
    get_LoaderClass,
)
from rag_retriever.modules.splitter.splitter import make_text_splitter
from rag_retriever.modules.splitter.zh_title_enhance import (
    zh_title_enhance as func_zh_title_enhance,
)

KB_ROOT_PATH = "./knowledge_base/"
TEXT_SPLITTER_NAME = "ChineseRecursiveTextSplitter"
ZH_TITLE_ENHANCE = False
CHUNK_SIZE = 250
OVERLAP_SIZE = 50


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


def get_file_path(knowledge_base_name: str, doc_name: str):
    doc_path = Path(get_doc_path(knowledge_base_name)).resolve()
    file_path = (doc_path / doc_name).resolve()
    if str(file_path).startswith(str(doc_path)):
        return str(file_path)


class KnowledgeFile:
    def __init__(
        self,
        filename: str,
        knowledge_base_name: str,
        loader_kwargs: dict = {},
    ):
        """
        对应知识库目录中的文件，必须是磁盘上存在的才能进行向量化等操作。
        """
        self.kb_name = knowledge_base_name
        self.filename = str(Path(filename).as_posix())
        self.ext = os.path.splitext(filename)[-1].lower()
        if self.ext not in SUPPORTED_EXTS:
            raise ValueError(f"暂未支持的文件格式 {self.filename}")
        self.loader_kwargs = loader_kwargs
        self.filepath = get_file_path(knowledge_base_name, filename)
        self.docs = None
        self.splited_docs = None
        self.document_loader_name = get_LoaderClass(self.ext)
        self.text_splitter_name = TEXT_SPLITTER_NAME

    def file2docs(self, refresh: bool = False):
        if self.docs is None or refresh:
            logger.info(f"{self.document_loader_name} used for {self.filepath}")
            loader = get_loader(
                loader_name=self.document_loader_name,
                file_path=self.filepath,
                loader_kwargs=self.loader_kwargs,
            )
            if isinstance(loader, TextLoader):
                loader.encoding = "utf8"
                self.docs = loader.load()
            else:
                self.docs = loader.load()
        return self.docs

    def docs2texts(
        self,
        docs: List[Document],
        zh_title_enhance: bool = ZH_TITLE_ENHANCE,
        refresh: bool = False,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        text_splitter: TextSplitter = None,
    ):
        docs = docs or self.file2docs(refresh=refresh)
        if not docs:
            return []
        if self.ext not in [".csv"]:
            if text_splitter is None:
                text_splitter = make_text_splitter(
                    splitter_name=self.text_splitter_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            if self.text_splitter_name == "MarkdownHeaderTextSplitter":
                docs = text_splitter.split_text(docs[0].page_content)
            else:
                docs = text_splitter.split_documents(docs)

        if not docs:
            return []

        print(f"文档切分示例：{docs[0]}")
        if zh_title_enhance:
            docs = func_zh_title_enhance(docs)
        self.splited_docs = docs
        return self.splited_docs

    def file2text(
        self,
        zh_title_enhance: bool = ZH_TITLE_ENHANCE,
        refresh: bool = False,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        text_splitter: TextSplitter = None,
    ):
        if self.splited_docs is None or refresh:
            docs = self.file2docs()
            self.splited_docs = self.docs2texts(
                docs=docs,
                zh_title_enhance=zh_title_enhance,
                refresh=refresh,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
        return self.splited_docs

    def file_exist(self):
        return os.path.isfile(self.filepath)

    def get_mtime(self):
        return os.path.getmtime(self.filepath)

    def get_size(self):
        return os.path.getsize(self.filepath)


def list_files_from_folder(kb_name: str):
    doc_path = get_doc_path(kb_name)
    result = []

    def is_skiped_path(path: str):
        tail = os.path.basename(path).lower()
        for x in ["temp", "tmp", ".", "~$"]:
            if tail.startswith(x):
                return True
        return False

    def process_entry(entry):
        if is_skiped_path(entry.path):
            return

        if entry.is_symlink():
            target_path = os.path.realpath(entry.path)
            with os.scandir(target_path) as target_it:
                for target_entry in target_it:
                    process_entry(target_entry)
        elif entry.is_file():
            file_path = Path(
                os.path.relpath(entry.path, doc_path)
            ).as_posix()  # 路径统一为 posix 格式
            result.append(file_path)
        elif entry.is_dir():
            with os.scandir(entry.path) as it:
                for sub_entry in it:
                    process_entry(sub_entry)

    with os.scandir(doc_path) as it:
        for entry in it:
            process_entry(entry)

    return result


def list_kbs_from_folder():
    return [
        f
        for f in os.listdir(KB_ROOT_PATH)
        if os.path.isdir(os.path.join(KB_ROOT_PATH, f))
    ]
