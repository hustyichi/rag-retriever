from rag_retriever.modules.doc_store.kb_service.base import KBService
from rag_retriever.modules.doc_store.kb_service.constants import SupportedVSType


def get_kb_service(
    kb_name: str,
    vector_store_type: SupportedVSType,
    embed_model: str,
    kb_info: str,
) -> KBService:
    params = {
        "knowledge_base_name": kb_name,
        "embed_model": embed_model,
        "kb_info": kb_info,
    }
    if SupportedVSType.FAISS == vector_store_type:
        from rag_retriever.modules.doc_store.kb_service.faiss_kb_service import (
            FaissKBService,
        )

        return FaissKBService(**params)
    elif SupportedVSType.PG == vector_store_type:
        from rag_retriever.modules.doc_store.kb_service.pg_kb_service import PGKBService

        return PGKBService(**params)
    elif SupportedVSType.RELYT == vector_store_type:
        from rag_retriever.modules.doc_store.kb_service.relyt_kb_service import (
            RelytKBService,
        )

        return RelytKBService(**params)
    elif SupportedVSType.MILVUS == vector_store_type:
        from rag_retriever.modules.doc_store.kb_service.milvus_kb_service import (
            MilvusKBService,
        )

        return MilvusKBService(**params)
    elif SupportedVSType.ZILLIZ == vector_store_type:
        from rag_retriever.modules.doc_store.kb_service.zilliz_kb_service import (
            ZillizKBService,
        )

        return ZillizKBService(**params)
    elif SupportedVSType.DEFAULT == vector_store_type:
        from rag_retriever.modules.doc_store.kb_service.milvus_kb_service import (
            MilvusKBService,
        )

        return MilvusKBService(
            **params
        )  # other milvus parameters are set in model_config.kbs_config
    elif SupportedVSType.ES == vector_store_type:
        from rag_retriever.modules.doc_store.kb_service.es_kb_service import ESKBService

        return ESKBService(**params)
    elif SupportedVSType.CHROMADB == vector_store_type:
        from rag_retriever.modules.doc_store.kb_service.chromadb_kb_service import (
            ChromaKBService,
        )

        return ChromaKBService(**params)
    else:
        raise ValueError(f"Unsupported vector store type {vector_store_type}")
