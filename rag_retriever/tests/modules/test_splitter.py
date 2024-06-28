from rag_retriever.modules.loaders import loaders
from rag_retriever.modules.splitter import splitter
from rag_retriever.tests import local_files


def test_splitter():
    first_local_file = local_files.get_test_files()[0]
    loader = loaders.get_loader_by_name(first_local_file)
    docs = loader.load()
    assert docs

    text_splitter = splitter.make_text_splitter("ChineseRecursiveTextSplitter")
    split_docs = text_splitter.split_documents(docs)
    assert split_docs
