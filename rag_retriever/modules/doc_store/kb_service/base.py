import operator
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.docstore.document import Document
from loguru import logger
from pydantic import BaseModel

from rag_retriever.modules.doc_store.embedding.utils import (
    KnowledgeFile,
    get_doc_path,
    get_Embeddings,
    get_kb_path,
    list_files_from_folder,
    list_kbs_from_folder,
)
from rag_retriever.modules.doc_store.kb_service.constants import SupportedVSType

# TODO: 修复 kbs_configs
kbs_config: dict[str, Any] = {}


def _check_embed_model(embed_model: str) -> bool:
    embeddings = get_Embeddings(embed_model=embed_model)
    try:
        embeddings.embed_query("this is a test")
        return True
    except Exception as e:
        logger.error(
            f"failed to access embed model '{embed_model}': {e}", exc_info=True
        )
        return False


class DocumentWithVSId(Document):
    """
    矢量化后的文档
    """

    id: str = None
    score: float = 3.0


class KnowledgeBaseSchema(BaseModel):
    id: int
    kb_name: str
    kb_info: Optional[str]
    vs_type: Optional[str]
    embed_model: Optional[str]
    file_count: Optional[int]
    create_time: Optional[datetime]

    class Config:
        from_attributes = True  # 确保可以从 ORM 实例进行验证


class KBService(ABC):
    def __init__(
        self,
        knowledge_base_name: str,
        kb_info: str,
        embed_model: str,
    ):
        self.kb_name = knowledge_base_name
        self.kb_info = kb_info
        self.embed_model = embed_model
        self.kb_path = get_kb_path(self.kb_name)
        self.doc_path = get_doc_path(self.kb_name)
        self.do_init()

    def __repr__(self) -> str:
        return f"{self.kb_name} @ {self.embed_model}"

    def save_vector_store(self):
        """
        保存向量库:FAISS保存到磁盘，milvus保存到数据库。PGVector暂未支持
        """
        pass

    def check_embed_model(self, error_msg: str) -> bool:
        if not _check_embed_model(self.embed_model):
            logger.error(error_msg, exc_info=True)
            return False
        else:
            return True

    def create_kb(self):
        """
        创建知识库
        """
        if not os.path.exists(self.doc_path):
            os.makedirs(self.doc_path)

        # TODO: 添加知识库到数据库
        # status = add_kb_to_db(
        #     self.kb_name, self.kb_info, self.vs_type(), self.embed_model
        # )

        # if status:
        #     self.do_create_kb()
        # return status

    def clear_vs(self):
        """
        删除向量库中所有内容
        """
        self.do_clear_vs()
        return True
        # TODO: delete files from db
        # status = delete_files_from_db(self.kb_name)
        # return status

    def drop_kb(self):
        """
        删除知识库
        """
        self.do_drop_kb()
        # TODO: 从数据库中删除知识库
        # status = delete_kb_from_db(self.kb_name)
        # return status

    def add_doc(self, kb_file: KnowledgeFile, docs: List[Document] = [], **kwargs):
        """
        向知识库添加文件
        如果指定了docs，则不再将文本向量化，并将数据库对应条目标为custom_docs=True
        """
        if not self.check_embed_model(
            f"could not add docs because failed to access embed model."
        ):
            return False

        if docs:
            custom_docs = True
        else:
            docs = kb_file.file2text()
            custom_docs = False

        if docs:
            # 将 metadata["source"] 改为相对路径
            for doc in docs:
                try:
                    doc.metadata.setdefault("source", kb_file.filename)
                    source = doc.metadata.get("source", "")
                    if os.path.isabs(source):
                        rel_path = Path(source).relative_to(self.doc_path)
                        doc.metadata["source"] = str(rel_path.as_posix().strip("/"))
                except Exception as e:
                    print(
                        f"cannot convert absolute path ({source}) to relative path. error is : {e}"
                    )
            self.delete_doc(kb_file)
            doc_infos = self.do_add_doc(docs, **kwargs)
            # TODO: fix add file to db
            # status = add_file_to_db(
            #     kb_file,
            #     custom_docs=custom_docs,
            #     docs_count=len(docs),
            #     doc_infos=doc_infos,
            # )
        else:
            status = False
        return status

    def delete_doc(
        self, kb_file: KnowledgeFile, delete_content: bool = False, **kwargs
    ):
        """
        从知识库删除文件
        """
        self.do_delete_doc(kb_file, **kwargs)
        # TODO: fix delete file from db
        # status = delete_file_from_db(kb_file)
        status = True
        if delete_content and os.path.exists(kb_file.filepath):
            os.remove(kb_file.filepath)
        return status

    def update_info(self, kb_info: str):
        """
        更新知识库介绍
        """
        self.kb_info = kb_info
        # TODO: 增加知识库
        # status = add_kb_to_db(
        #     self.kb_name, self.kb_info, self.vs_type(), self.embed_model
        # )
        # return status

    def update_doc(self, kb_file: KnowledgeFile, docs: List[Document] = [], **kwargs):
        """
        使用content中的文件更新向量库
        如果指定了docs，则使用自定义docs，并将数据库对应条目标为custom_docs=True
        """
        if not self.check_embed_model(
            f"could not update docs because failed to access embed model."
        ):
            return False

        if os.path.exists(kb_file.filepath):
            self.delete_doc(kb_file, **kwargs)
            return self.add_doc(kb_file, docs=docs, **kwargs)

    def exist_doc(self, file_name: str):
        # TODO: fix exist doc
        return False
        # return file_exists_in_db(
        #     KnowledgeFile(knowledge_base_name=self.kb_name, filename=file_name)
        # )

    def list_files(self):
        # TODO: fix files
        return []
        # return list_files_from_db(self.kb_name)

    def count_files(self):
        # TODO: fix connt files
        return 0
        # return count_files_from_db(self.kb_name)

    def search_docs(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
    ) -> List[Document]:
        if not self.check_embed_model(
            f"could not search docs because failed to access embed model."
        ):
            return []
        docs = self.do_search(query, top_k, score_threshold)
        return docs

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        return []

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        raise NotImplementedError

    def update_doc_by_ids(self, docs: Dict[str, Document]) -> bool:
        """
        传入参数为： {doc_id: Document, ...}
        如果对应 doc_id 的值为 None，或其 page_content 为空，则删除该文档
        """
        if not self.check_embed_model(
            f"could not update docs because failed to access embed model."
        ):
            return False

        self.del_doc_by_ids(list(docs.keys()))
        pending_docs = []
        ids = []
        for _id, doc in docs.items():
            if not doc or not doc.page_content.strip():
                continue
            ids.append(_id)
            pending_docs.append(doc)
        self.do_add_doc(docs=pending_docs, ids=ids)
        return True

    def list_docs(
        self, file_name: str = None, metadata: Dict = {}
    ) -> List[DocumentWithVSId]:
        """
        通过file_name或metadata检索Document
        """
        # TODO: fix doc infos
        doc_infos = []
        # doc_infos = list_docs_from_db(
        #     kb_name=self.kb_name, file_name=file_name, metadata=metadata
        # )
        docs = []
        for x in doc_infos:
            doc_info = self.get_doc_by_ids([x["id"]])[0]
            if doc_info is not None:
                # 处理非空的情况
                doc_with_id = DocumentWithVSId(**doc_info.dict(), id=x["id"])
                docs.append(doc_with_id)
            else:
                # 处理空的情况
                # 可以选择跳过当前循环迭代或执行其他操作
                pass
        return docs

    def get_relative_source_path(self, filepath: str):
        """
        将文件路径转化为相对路径，保证查询时一致
        """
        relative_path = filepath
        if os.path.isabs(relative_path):
            try:
                relative_path = Path(filepath).relative_to(self.doc_path)
            except Exception as e:
                print(
                    f"cannot convert absolute path ({relative_path}) to relative path. error is : {e}"
                )

        relative_path = str(relative_path.as_posix().strip("/"))
        return relative_path

    @abstractmethod
    def do_create_kb(self):
        """
        创建知识库子类实自己逻辑
        """
        pass

    @staticmethod
    def list_kbs_type():
        return list(kbs_config.keys())

    @classmethod
    def list_kbs(cls):
        # TODO: fix this
        return []
        # return list_kbs_from_db()

    def exists(self, kb_name: str = None):
        kb_name = kb_name or self.kb_name
        # TODO: 支持判断知识库是否存在
        return False
        # return kb_exists(kb_name)

    @abstractmethod
    def vs_type(self) -> str:
        pass

    @abstractmethod
    def do_init(self):
        pass

    @abstractmethod
    def do_drop_kb(self):
        """
        删除知识库子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_search(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
    ) -> List[Tuple[Document, float]]:
        """
        搜索知识库子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_add_doc(
        self,
        docs: List[Document],
        **kwargs,
    ) -> List[Dict]:
        """
        向知识库添加文档子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_delete_doc(self, kb_file: KnowledgeFile):
        """
        从知识库删除文档子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_clear_vs(self):
        """
        从知识库删除全部向量子类实自己逻辑
        """
        pass


def get_kb_details() -> List[Dict]:
    kbs_in_folder = list_kbs_from_folder()
    kbs_in_db: List[KnowledgeBaseSchema] = KBService.list_kbs()
    result = {}

    for kb in kbs_in_folder:
        result[kb] = {
            "kb_name": kb,
            "vs_type": "",
            "kb_info": "",
            "embed_model": "",
            "file_count": 0,
            "create_time": None,
            "in_folder": True,
            "in_db": False,
        }

    for kb_detail in kbs_in_db:
        kb_detail = kb_detail.model_dump()
        kb_name = kb_detail["kb_name"]
        kb_detail["in_db"] = True
        if kb_name in result:
            result[kb_name].update(kb_detail)
        else:
            kb_detail["in_folder"] = False
            result[kb_name] = kb_detail

    data = []
    for i, v in enumerate(result.values()):
        v["No"] = i + 1
        data.append(v)

    return data


def score_threshold_process(score_threshold, k, docs):
    if score_threshold is not None:
        cmp = operator.le
        docs = [
            (doc, similarity)
            for doc, similarity in docs
            if cmp(similarity, score_threshold)
        ]
    return docs[:k]
