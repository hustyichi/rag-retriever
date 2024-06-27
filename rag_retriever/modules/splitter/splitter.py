import importlib
from typing import Any

from langchain.text_splitter import MarkdownHeaderTextSplitter

text_splitter_dict: dict[str, Any] = {
    "ChineseRecursiveTextSplitter": {
        "source": "huggingface",  # 选择tiktoken则使用openai的方法
        "tokenizer_name_or_path": "",  # 需要设置对应的 tokenizer 的地址
    },
    "SpacyTextSplitter": {
        "source": "huggingface",
        "tokenizer_name_or_path": "gpt2",
    },
    "RecursiveCharacterTextSplitter": {
        "source": "tiktoken",
        "tokenizer_name_or_path": "cl100k_base",
    },
    "MarkdownHeaderTextSplitter": {
        "headers_to_split_on": [
            ("#", "head1"),
            ("##", "head2"),
            ("###", "head3"),
            ("####", "head4"),
        ]
    },
}


def make_text_splitter(
    splitter_name: str, chunk_size: int = 250, chunk_overlap: int = 50
):
    """
    根据参数获取特定的分词器
    """
    splitter_name = splitter_name or "SpacyTextSplitter"
    try:
        if (
            splitter_name == "MarkdownHeaderTextSplitter"
        ):  # MarkdownHeaderTextSplitter特殊判定
            headers_to_split_on = text_splitter_dict[splitter_name][
                "headers_to_split_on"
            ]
            text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, strip_headers=False
            )
        else:
            try:  # 优先使用用户自定义的text_splitter
                text_splitter_module = importlib.import_module(
                    "rag_retriever.modules.splitter.custom_splitter"
                )
                TextSplitter = getattr(text_splitter_module, splitter_name)
            except:  # 否则使用langchain的text_splitter
                text_splitter_module = importlib.import_module(
                    "langchain.text_splitter"
                )
                TextSplitter = getattr(text_splitter_module, splitter_name)

            if (
                text_splitter_dict[splitter_name]["source"] == "tiktoken"
            ):  # 从tiktoken加载
                try:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name][
                            "tokenizer_name_or_path"
                        ],
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                except:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name][
                            "tokenizer_name_or_path"
                        ],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
            elif (
                text_splitter_dict[splitter_name]["source"] == "huggingface"
            ):  # 从huggingface加载
                if (
                    text_splitter_dict[splitter_name]["tokenizer_name_or_path"]
                    == "gpt2"
                ):
                    from langchain.text_splitter import CharacterTextSplitter
                    from transformers import GPT2TokenizerFast

                    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                else:  # 字符长度加载
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        trust_remote_code=True,
                    )
                text_splitter = TextSplitter.from_huggingface_tokenizer(
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            else:
                try:
                    text_splitter = TextSplitter(
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                except:
                    text_splitter = TextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
    except Exception as e:
        print(e)
        text_splitter_module = importlib.import_module("langchain.text_splitter")
        TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
        text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # If you use SpacyTextSplitter you can use GPU to do split likes Issue #1287
    # text_splitter._tokenizer.max_length = 37016792
    # text_splitter._tokenizer.prefer_gpu()
    return text_splitter
