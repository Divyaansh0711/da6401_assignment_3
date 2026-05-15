from collections import Counter
from typing import List, Tuple, Optional
import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import spacy
from datasets import load_dataset


SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


class Vocabulary:
    """
    Simple vocabulary class without torchtext dependency.

    Required behavior:
    - token -> index using stoi
    - index -> token using itos
    - unknown tokens map to <unk>
    - supports lookup_token(idx), lookup_index(token)
    """

    def __init__(self, min_freq: int = 2) -> None:
        self.min_freq = min_freq
        self.stoi = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.itos = list(SPECIAL_TOKENS)

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.lookup_index(token)

    def lookup_index(self, token: str) -> int:
        return self.stoi.get(token, UNK_IDX)

    def lookup_token(self, index: int) -> str:
        return self.itos[index]

    def build_from_tokens(self, tokenized_sentences: List[List[str]]) -> None:
        counter = Counter()

        for tokens in tokenized_sentences:
            counter.update(tokens)

        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def encode(self, tokens: List[str], add_sos_eos: bool = True) -> List[int]:
        indices = [self.lookup_index(token) for token in tokens]

        if add_sos_eos:
            indices = [SOS_IDX] + indices + [EOS_IDX]

        return indices

    def decode(self, indices: List[int], remove_specials: bool = True) -> List[str]:
        tokens = []

        for idx in indices:
            token = self.lookup_token(int(idx))

            if remove_specials and token in SPECIAL_TOKENS:
                continue

            tokens.append(token)

        return tokens

    def to_dict(self) -> dict:
        return {
            "min_freq": self.min_freq,
            "itos": self.itos,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        vocab = cls(min_freq=data.get("min_freq", 2))
        vocab.itos = data["itos"]
        vocab.stoi = {token: idx for idx, token in enumerate(vocab.itos)}
        return vocab


class Multi30kDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[Vocabulary] = None,
        tgt_vocab: Optional[Vocabulary] = None,
        min_freq: int = 2,
        max_len: int = 100,
    ):
        """
        Loads the Multi30k dataset and prepares tokenizers.

        Source language: German
        Target language: English
        """
        self.split = split
        self.min_freq = min_freq
        self.max_len = max_len

        self.raw_dataset = load_dataset(
            "bentrevett/multi30k",
            split=split,
            trust_remote_code=True,
        )

        self.spacy_de = self._load_spacy_tokenizer("de")
        self.spacy_en = self._load_spacy_tokenizer("en")

        self.src_sentences = []
        self.tgt_sentences = []

        for example in self.raw_dataset:
            de_sentence, en_sentence = self._extract_pair(example)
            self.src_sentences.append(de_sentence)
            self.tgt_sentences.append(en_sentence)

        self.tokenized_src = [self.tokenize_de(sentence) for sentence in self.src_sentences]
        self.tokenized_tgt = [self.tokenize_en(sentence) for sentence in self.tgt_sentences]

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        if self.src_vocab is None or self.tgt_vocab is None:
            self.build_vocab()

        self.data = self.process_data()

    def _load_spacy_tokenizer(self, lang: str):
        """
        Loads spaCy tokenizer. Falls back to blank tokenizer if small model is not installed.
        """
        try:
            if lang == "de":
                return spacy.load("de_core_news_sm")
            if lang == "en":
                return spacy.load("en_core_web_sm")
        except OSError:
            return spacy.blank(lang)

        return spacy.blank(lang)

    def _extract_pair(self, example: dict) -> Tuple[str, str]:
        """
        Handles possible dataset formats safely.
        """
        if "de" in example and "en" in example:
            return example["de"], example["en"]

        if "translation" in example:
            translation = example["translation"]
            return translation["de"], translation["en"]

        raise KeyError(f"Could not find German-English fields in example keys: {example.keys()}")

    def tokenize_de(self, text: str) -> List[str]:
        return [token.text.lower() for token in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text: str) -> List[str]:
        return [token.text.lower() for token in self.spacy_en.tokenizer(text)]

    def build_vocab(self):
        """
        Builds source and target vocabularies with:
        <unk>, <pad>, <sos>, <eos>
        """
        if self.src_vocab is None:
            self.src_vocab = Vocabulary(min_freq=self.min_freq)
            self.src_vocab.build_from_tokens(self.tokenized_src)

        if self.tgt_vocab is None:
            self.tgt_vocab = Vocabulary(min_freq=self.min_freq)
            self.tgt_vocab.build_from_tokens(self.tokenized_tgt)

        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        """
        Converts German and English sentences into integer token lists.
        """
        processed = []

        for src_tokens, tgt_tokens in zip(self.tokenized_src, self.tokenized_tgt):
            src_indices = self.src_vocab.encode(src_tokens, add_sos_eos=True)
            tgt_indices = self.tgt_vocab.encode(tgt_tokens, add_sos_eos=True)

            if len(src_indices) <= self.max_len and len(tgt_indices) <= self.max_len:
                processed.append(
                    (
                        torch.tensor(src_indices, dtype=torch.long),
                        torch.tensor(tgt_indices, dtype=torch.long),
                    )
                )

        return processed

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

    @staticmethod
    def collate_fn(batch):
        """
        Pads src and tgt batches.

        Returns:
            src_batch: [batch, src_len]
            tgt_batch: [batch, tgt_len]
        """
        src_batch, tgt_batch = zip(*batch)

        src_batch = pad_sequence(
            src_batch,
            batch_first=True,
            padding_value=PAD_IDX,
        )

        tgt_batch = pad_sequence(
            tgt_batch,
            batch_first=True,
            padding_value=PAD_IDX,
        )

        return src_batch, tgt_batch

    def save_vocabs(self, path: str = "vocabs.json") -> None:
        """
        Saves source and target vocabularies for inference.
        """
        data = {
            "src_vocab": self.src_vocab.to_dict(),
            "tgt_vocab": self.tgt_vocab.to_dict(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_vocabs(path: str = "vocabs.json") -> Tuple[Vocabulary, Vocabulary]:
        """
        Loads source and target vocabularies.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        src_vocab = Vocabulary.from_dict(data["src_vocab"])
        tgt_vocab = Vocabulary.from_dict(data["tgt_vocab"])

        return src_vocab, tgt_vocab