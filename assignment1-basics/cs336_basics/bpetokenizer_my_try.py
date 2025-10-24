from collections import defaultdict, Counter
import regex as re
from typing import Optional, Iterable

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        spectial_token: Optional[list[str]] = None  
    ):
        pass

    def encode(self, text: str) -> list[int]:
        """
        将一个原始字符串编码为一个 token ID 列表。
        """
        return []
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        将一个字符串迭代器（如文件）编码为 token ID 迭代器。
        """
        pass


    def decode(self, ids: list[int]) -> str:
        """
        将一个 token ID 列表解码回一个字符串。
        """
        return ""
    