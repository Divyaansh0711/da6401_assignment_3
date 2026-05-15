"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
from typing import Optional, Tuple, List

import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import spacy
except ImportError:
    spacy = None

try:
    from dataset import (
        Multi30kDataset,
        PAD_IDX,
        SOS_IDX,
        EOS_IDX,
        Vocabulary,
    )
except ImportError:
    PAD_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3
    Multi30kDataset = None
    Vocabulary = None


# Fill these after training and uploading files to Google Drive.
CHECKPOINT_GDRIVE_ID = "1XCP8S-A6buXo0aWTC13nYumhATWYX_mA"
VOCAB_GDRIVE_ID = "1Bi9f3PeapZX5M-eCtT6eFAc9k89BmiWt"

DEFAULT_CHECKPOINT_PATH = "/tmp/checkpoint.pt"
DEFAULT_VOCAB_PATH = "/tmp/vocabs.json"


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

    Args:
        Q    : shape (..., seq_q, d_k)
        K    : shape (..., seq_k, d_k)
        V    : shape (..., seq_k, d_v)
        mask : bool mask broadcastable to (..., seq_q, seq_k)
               True means masked out.

    Returns:
        output : shape (..., seq_q, d_v)
        attn_w : shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_w = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)

    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#   MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Source padding mask.

    Args:
        src: [batch, src_len]

    Returns:
        [batch, 1, 1, src_len]
        True means masked out.
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Combined target padding + causal mask.

    Args:
        tgt: [batch, tgt_len]

    Returns:
        [batch, 1, tgt_len, tgt_len]
        True means masked out.
    """
    _, tgt_len = tgt.shape

    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool),
        diagonal=1,
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#   MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        self.attn_weights = attn_weights

        attn_output = self.dropout(attn_output)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        return self.W_o(attn_output)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#   FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#   ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Transformer encoder layer using Pre-LayerNorm.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, x_norm, x_norm, src_mask)
        x = x + self.dropout1(attn_out)

        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_out)

        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Transformer decoder layer using Pre-LayerNorm.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        self_attn_out = self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
        x = x + self.dropout1(self_attn_out)

        x_norm = self.norm2(x)
        cross_attn_out = self.cross_attn(x_norm, memory, memory, src_mask)
        x = x + self.dropout2(cross_attn_out)

        ff_out = self.feed_forward(self.norm3(x))
        x = x + self.dropout3(ff_out)

        return x


# ══════════════════════════════════════════════════════════════════════
#   ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """
    Stack of N encoder layers.
    """

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class Decoder(nn.Module):
    """
    Stack of N decoder layers.
    """

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for German-to-English translation.
    """

    def __init__(
        self,
        src_vocab_size: int = 10000,
        tgt_vocab_size: int = 10000,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: str = None,
        vocab_path: str = DEFAULT_VOCAB_PATH,
        max_len: int = 100,
        load_checkpoint_by_default: bool = False,
    ) -> None:
        super().__init__()

        self.vocab_path = vocab_path
        self.max_len = max_len

        self.src_vocab = None
        self.tgt_vocab = None
        self.spacy_de = None

        self._maybe_download_vocab(vocab_path)
        self._load_vocab_if_available(vocab_path)

        if self.src_vocab is not None and self.tgt_vocab is not None:
            src_vocab_size = len(self.src_vocab)
            tgt_vocab_size = len(self.tgt_vocab)

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

        self.encoder = Encoder(encoder_layer, N)
        self.decoder = Decoder(decoder_layer, N)

        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

        if checkpoint_path is None and load_checkpoint_by_default:
            checkpoint_path = DEFAULT_CHECKPOINT_PATH

        if checkpoint_path is not None:
            self._load_checkpoint_from_path_or_drive(checkpoint_path)

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _maybe_download_vocab(self, vocab_path: str) -> None:
        if os.path.exists(vocab_path):
            return

        if VOCAB_GDRIVE_ID:
            gdown.download(id=VOCAB_GDRIVE_ID, output=vocab_path, quiet=False)

    def _load_vocab_if_available(self, vocab_path: str) -> None:
        if Multi30kDataset is None:
            return

        if not os.path.exists(vocab_path):
            return

        self.src_vocab, self.tgt_vocab = Multi30kDataset.load_vocabs(vocab_path)

    def _load_checkpoint_from_path_or_drive(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            if CHECKPOINT_GDRIVE_ID:
                gdown.download(id=CHECKPOINT_GDRIVE_ID, output=checkpoint_path, quiet=False)

        if not os.path.exists(checkpoint_path):
            return

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.load_state_dict(checkpoint)

    def _load_spacy_de(self):
        if self.spacy_de is not None:
            return self.spacy_de

        if spacy is None:
            raise ImportError("spacy is required for Transformer.infer()")

        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            self.spacy_de = spacy.blank("de")

        return self.spacy_de

    def _tokenize_de(self, text: str) -> List[str]:
        tokenizer = self._load_spacy_de().tokenizer
        return [token.text.lower() for token in tokenizer(text)]

    def _decode_target_tokens(self, indices: List[int]) -> str:
        if self.tgt_vocab is None:
            raise RuntimeError("Target vocabulary is not loaded. Make sure vocabs.json is available.")

        tokens = []

        for idx in indices:
            token = self.tgt_vocab.lookup_token(int(idx))

            if token == "<eos>":
                break

            if token in {"<unk>", "<pad>", "<sos>", "<eos>"}:
                continue

            tokens.append(token)

        sentence = " ".join(tokens)

        sentence = sentence.replace(" .", ".")
        sentence = sentence.replace(" ,", ",")
        sentence = sentence.replace(" !", "!")
        sentence = sentence.replace(" ?", "?")
        sentence = sentence.replace(" '", "'")
        sentence = sentence.replace(" n't", "n't")

        return sentence.strip()

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)

        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)

        decoder_out = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        logits = self.generator(decoder_out)

        return logits

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        logits = self.decode(memory, src_mask, tgt, tgt_mask)

        return logits

    def infer(self, src_sentence: str) -> str:
        """
        Translate a single German sentence to English using greedy decoding.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            self._maybe_download_vocab(self.vocab_path)
            self._load_vocab_if_available(self.vocab_path)

        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError(
                "Vocabulary not loaded. Train the model first and make sure vocabs.json is available."
            )

        device = next(self.parameters()).device

        src_tokens = self._tokenize_de(src_sentence)
        src_indices = [SOS_IDX] + [self.src_vocab.lookup_index(tok) for tok in src_tokens] + [EOS_IDX]

        src = torch.tensor(src_indices, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src, PAD_IDX).to(device)

        self.eval()

        with torch.no_grad():
            memory = self.encode(src, src_mask)

            ys = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)

            for _ in range(self.max_len - 1):
                tgt_mask = make_tgt_mask(ys, PAD_IDX).to(device)
                logits = self.decode(memory, src_mask, ys, tgt_mask)

                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)

                ys = torch.cat([ys, next_token_tensor], dim=1)

                if next_token == EOS_IDX:
                    break

        return self._decode_target_tokens(ys.squeeze(0).tolist())