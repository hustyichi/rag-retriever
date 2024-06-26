from typing import Dict, List, Union

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class CustomHTMLLoader(BaseLoader):
    """Custom Load `HTML` files and parse them with `beautiful soup`."""

    def __init__(
        self,
        file_path: str,
        open_encoding: Union[str, None] = None,
        bs_kwargs: Union[dict, None] = None,
        get_text_separator: str = "",
    ) -> None:
        """initialize with path, and optionally, file encoding to use, and any kwargs
        to pass to the BeautifulSoup object.

        Args:
            file_path: The path to the file to load.
            open_encoding: The encoding to use when opening the file.
            bs_kwargs: Any kwargs to pass to the BeautifulSoup object.
            get_text_separator: The separator to use when calling get_text on the soup.
        """
        try:
            import bs4  # noqa:F401
        except ImportError:
            raise ImportError(
                "beautifulsoup4 package not found, please install it with "
                "`pip install beautifulsoup4`"
            )

        self.file_path = file_path
        self.open_encoding = open_encoding
        if bs_kwargs is None:
            bs_kwargs = {"features": "lxml"}
        self.bs_kwargs = bs_kwargs
        self.get_text_separator = get_text_separator

    def load(self) -> List[Document]:
        """Load HTML document into document objects."""
        from bs4 import BeautifulSoup

        with open(self.file_path, "r", encoding=self.open_encoding) as f:
            soup = BeautifulSoup(f, **self.bs_kwargs)

        # clean unchecked checkboxes and radio buttons
        match_inputs = soup.find_all("input", {"type": ["checkbox", "radio"]})
        for input_tag in match_inputs:
            if not input_tag.has_attr("checked"):
                next_span = input_tag.find_next_sibling("span")
                if next_span:
                    next_span.extract()
                input_tag.extract()

        text = soup.get_text(self.get_text_separator)

        if soup.title:
            title = str(soup.title.string)
        else:
            title = ""

        metadata: Dict[str, Union[str, None]] = {
            "source": self.file_path,
            "title": title,
        }
        return [Document(page_content=text, metadata=metadata)]
