from rag_retriever.modules.retriever.custom_retriever import (
    BaseRetrieverService,
    EnsembleRetrieverService,
    VectorstoreRetrieverService,
)

Retrivals = {
    "vectorstore": VectorstoreRetrieverService,
    "ensemble": EnsembleRetrieverService,
}


def get_Retriever(type: str = "vectorstore") -> BaseRetrieverService:
    return Retrivals[type]
