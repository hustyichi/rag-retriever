from rag_retriever.modules.loaders import loaders
from rag_retriever.tests import local_files


def test_loader():
    first_local_file = local_files.get_test_files()[0]
    loader = loaders.get_loader_by_name(first_local_file)
    data = loader.load()
    assert data
