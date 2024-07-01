from typing import Any, Optional, Sequence

import torch
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from pydantic import Field, PrivateAttr
from sentence_transformers import CrossEncoder


class LangchainReranker(BaseDocumentCompressor):
    model_name_or_path: str = Field()
    _model: Any = PrivateAttr()
    top_n: int = Field()
    device: str = Field()
    max_length: int = Field()
    batch_size: int = Field()
    num_workers: int = Field()
    score_threshold: float = Field()

    def __init__(
        self,
        model_name_or_path: str,
        top_n: int = 3,
        device: str = "cuda",
        max_length: int = 1024,
        batch_size: int = 32,
        num_workers: int = 0,
        score_threshold: float = 0.1,
    ):

        self._model = CrossEncoder(
            model_name=model_name_or_path, max_length=max_length, device=device
        )
        super().__init__(
            top_n=top_n,
            model_name_or_path=model_name_or_path,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers,
            score_threshold=score_threshold,
        )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if len(documents) == 0:
            return []

        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        sentence_pairs = [[query, _doc] for _doc in _docs]
        results = self._model.predict(
            sentences=sentence_pairs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            convert_to_tensor=True,
        )

        values, indices = results.topk(len(results))

        final_results: list[Document] = []
        for value, index in zip(values, indices):
            score = value.item() if isinstance(value, torch.Tensor) else value
            if len(final_results) >= self.top_n or score < self.score_threshold:
                break

            doc = doc_list[index]
            doc.metadata["relevance_score"] = score
            final_results.append(doc)
        return final_results
