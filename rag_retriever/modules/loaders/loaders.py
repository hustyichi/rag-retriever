import importlib
import json
import os
from typing import Optional

import chardet
import langchain_community
from langchain_community.document_loaders import JSONLoader
from loguru import logger

LOADER_DICT = {
    "UnstructuredHTMLLoader": [".html", ".htm"],
    "MHTMLLoader": [".mhtml"],
    "TextLoader": [".md"],
    "UnstructuredMarkdownLoader": [".md"],
    "JSONLoader": [".json"],
    "JSONLinesLoader": [".jsonl"],
    "CSVLoader": [".csv"],
    # "FilteredCSVLoader": [".csv"], 如果使用自定义分割csv
    "RapidOCRPDFLoader": [".pdf"],
    "RapidOCRDocLoader": [".docx", ".doc"],
    "RapidOCRPPTLoader": [
        ".ppt",
        ".pptx",
    ],
    "RapidOCRLoader": [".png", ".jpg", ".jpeg", ".bmp"],
    "UnstructuredFileLoader": [
        ".eml",
        ".msg",
        ".rst",
        ".rtf",
        ".txt",
        ".xml",
        ".epub",
        ".odt",
        ".tsv",
    ],
    "UnstructuredEmailLoader": [".eml", ".msg"],
    "UnstructuredEPubLoader": [".epub"],
    "UnstructuredExcelLoader": [".xlsx", ".xls", ".xlsd"],
    "NotebookLoader": [".ipynb"],
    "UnstructuredODTLoader": [".odt"],
    "PythonLoader": [".py"],
    "UnstructuredRSTLoader": [".rst"],
    "UnstructuredRTFLoader": [".rtf"],
    "SRTLoader": [".srt"],
    "TomlLoader": [".toml"],
    "UnstructuredTSVLoader": [".tsv"],
    "UnstructuredWordDocumentLoader": [".docx", ".doc"],
    "UnstructuredXMLLoader": [".xml"],
    "UnstructuredPowerPointLoader": [".ppt", ".pptx"],
    "EverNoteLoader": [".enex"],
}
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]


# patch json.dumps to disable ensure_ascii
def _new_json_dumps(obj, **kwargs):
    kwargs["ensure_ascii"] = False
    return _origin_json_dumps(obj, **kwargs)


if json.dumps is not _new_json_dumps:
    _origin_json_dumps = json.dumps
    json.dumps = _new_json_dumps


class JSONLinesLoader(JSONLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._json_lines = True


langchain_community.document_loaders.JSONLinesLoader = JSONLinesLoader


def get_LoaderClass(file_extension):
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            return LoaderClass


# 指定对应的加载器名选择对应的加载器
def get_loader(loader_name: str, file_path: str, loader_kwargs: Optional[dict] = None):
    """
    根据loader_name和文件路径或内容返回文档加载器。
    """
    loader_kwargs = loader_kwargs or {}
    try:
        if loader_name in [
            "RapidOCRPDFLoader",
            "RapidOCRLoader",
            "FilteredCSVLoader",
            "RapidOCRDocLoader",
            "RapidOCRPPTLoader",
            "CustomHTMLLoader",
        ]:
            document_loaders_module = importlib.import_module(
                "rag_retriever.modules.loaders.custom_loaders"
            )
        else:
            document_loaders_module = importlib.import_module(
                "langchain_community.document_loaders"
            )
        DocumentLoader = getattr(document_loaders_module, loader_name)
    except Exception as e:
        msg = f"为文件{file_path}查找加载器{loader_name}时出错：{e}"
        logger.error(f"{e.__class__.__name__}: {msg}", exc_info=e)

        document_loaders_module = importlib.import_module(
            "langchain_community.document_loaders"
        )
        DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

    if loader_name == "UnstructuredFileLoader":
        loader_kwargs.setdefault("autodetect_encoding", True)
    elif loader_name == "CSVLoader":
        if not loader_kwargs.get("encoding"):
            # 如果未指定 encoding，自动识别文件编码类型，避免langchain loader 加载文件报编码错误
            with open(file_path, "rb") as struct_file:
                encode_detect = chardet.detect(struct_file.read())
            if encode_detect is None:
                encode_detect = {"encoding": "utf-8"}
            loader_kwargs["encoding"] = encode_detect["encoding"]

    elif loader_name == "JSONLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)
    elif loader_name == "JSONLinesLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)

    loader = DocumentLoader(file_path, **loader_kwargs)
    return loader


# 根据文件名选择对应的加载器
def get_loader_by_name(file_path: str, loader_kwargs: Optional[dict] = None):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"暂未支持的文件格式 {file_path}")

    loader_name = get_LoaderClass(ext)
    loader = get_loader(
        loader_name=loader_name,
        file_path=file_path,
        loader_kwargs=loader_kwargs,
    )

    return loader
