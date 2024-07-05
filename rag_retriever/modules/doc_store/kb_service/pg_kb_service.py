import json
import shutil
from typing import Any, Dict, List, Optional

import sqlalchemy
from langchain.schema import Document
from langchain.vectorstores.pgvector import DistanceStrategy, PGVector
from sqlalchemy import text
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session

from rag_retriever.modules.doc_store.embedding.utils import (
    KnowledgeFile,
    get_Embeddings,
)
from rag_retriever.modules.doc_store.kb_service.base import KBService
from rag_retriever.modules.doc_store.kb_service.constants import SupportedVSType
from rag_retriever.modules.retriever.retriever import get_Retriever

# TODO: 支持 kbs_config
kbs_config: dict[str, Any] = {}


class PGKBService(KBService):
    engine: Engine = sqlalchemy.create_engine(
        kbs_config.get("pg").get("connection_uri"), pool_size=10
    )

    def _load_pg_vector(self):
        self.pg_vector = PGVector(
            embedding_function=get_Embeddings(self.embed_model),
            collection_name=self.kb_name,
            distance_strategy=DistanceStrategy.EUCLIDEAN,
            connection=PGKBService.engine,
            connection_string=kbs_config.get("pg").get("connection_uri"),
        )

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        with Session(PGKBService.engine) as session:
            stmt = text(
                "SELECT document, cmetadata FROM langchain_pg_embedding WHERE custom_id = ANY(:ids)"
            )
            results = [
                Document(page_content=row[0], metadata=row[1])
                for row in session.execute(stmt, {"ids": ids}).fetchall()
            ]
            return results

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        return super().del_doc_by_ids(ids)

    def do_init(self):
        self._load_pg_vector()

    def do_create_kb(self):
        pass

    def vs_type(self) -> str:
        return SupportedVSType.PG

    def do_drop_kb(self):
        with Session(PGKBService.engine) as session:
            session.execute(
                text(
                    f"""
                    -- 删除 langchain_pg_embedding 表中关联到 langchain_pg_collection 表中 的记录
                    DELETE FROM langchain_pg_embedding
                    WHERE collection_id IN (
                      SELECT uuid FROM langchain_pg_collection WHERE name = '{self.kb_name}'
                    );
                    -- 删除 langchain_pg_collection 表中 记录
                    DELETE FROM langchain_pg_collection WHERE name = '{self.kb_name}';
            """
                )
            )
            session.commit()
            shutil.rmtree(self.kb_path)

    def do_search(self, query: str, top_k: int, score_threshold: float):
        retriever = get_Retriever("vectorstore").from_vectorstore(
            self.pg_vector,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        docs = retriever.get_relevant_documents(query)
        return docs

    def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
        ids = self.pg_vector.add_documents(docs)
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        return doc_infos

    def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
        with Session(PGKBService.engine) as session:
            session.execute(
                text(
                    """ DELETE FROM langchain_pg_embedding WHERE cmetadata::jsonb @> '{"source": "filepath"}'::jsonb;""".replace(
                        "filepath", self.get_relative_source_path(kb_file.filepath)
                    )
                )
            )
            session.commit()

    def do_clear_vs(self):
        self.pg_vector.delete_collection()
        self.pg_vector.create_collection()
