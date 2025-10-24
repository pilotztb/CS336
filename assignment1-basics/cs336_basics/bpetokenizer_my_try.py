from collections import defaultdict, Counter
import regex as re

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        spectial_tokens: list[str]
    ):
        self.vocab = vocab
        self.merges = merges
        self.spectialTokens = spectial_tokens or []
        

        self.bytesToIntVocab = {v : k for k, v in vocab.items()}

        self.bpeRanks = dict(zip(merges, range(len(merges))))

        self.spectialTokenBytes = [token.encode("utf-8") for token in self.spectialTokens]

        for tokenBytes in self.spectialTokenBytes:
            if tokenBytes not in self.bytesToIntVocab:
                newID = len(self.vocab)
                self.vocab[newID] = tokenBytes
                self.bytesToIntVocab[tokenBytes] = newID



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
    