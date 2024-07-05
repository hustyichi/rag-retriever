from enum import Enum


class SupportedVSType(str, Enum):
    FAISS = "faiss"
    MILVUS = "milvus"
    DEFAULT = "default"
    ZILLIZ = "zilliz"
    PG = "pg"
    RELYT = "relyt"
    ES = "es"
    CHROMADB = "chromadb"
