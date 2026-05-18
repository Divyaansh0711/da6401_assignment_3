"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

Includes:
- Label smoothing
- Noam scheduler / fixed LR comparison
- Scaling-factor ablation
- Sinusoidal vs learned positional encoding
- Prediction confidence logging
- Q/K gradient norm logging
- Attention heatmap logging
- BLEU evaluation
"""

from collections import Counter
from typing import Optional, List, Dict, Tuple
import math
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

import model as model_module
from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from dataset import Multi30kDataset, PAD_IDX, SOS_IDX, EOS_IDX


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Args:
        vocab_size (int): Number of output classes.
        pad_idx (int): Index of <pad> token.
        smoothing (float): Smoothing factor epsilon.
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()

        assert 0.0 <= smoothing < 1.0, "smoothing must be in [0, 1)"

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch * tgt_len, vocab_size]
            target: [batch * tgt_len]
        """
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)

            if self.vocab_size > 2:
                smooth_value = self.smoothing / (self.vocab_size - 2)
            else:
                smooth_value = 0.0

            true_dist.fill_(smooth_value)
            true_dist[:, self.pad_idx] = 0.0

            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

            pad_mask = target.eq(self.pad_idx)
            true_dist[pad_mask] = 0.0

        loss = -(true_dist * log_probs).sum(dim=-1)
        non_pad_tokens = target.ne(self.pad_idx).sum().clamp_min(1)

        return loss.sum() / non_pad_tokens


# ══════════════════════════════════════════════════════════════════════
#  LEARNED POSITIONAL ENCODING FOR ABLATION
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding ablation using nn.Embedding.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()

        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)

        x = x + self.position_embedding(positions)

        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  ATTENTION SCALING ABLATION
# ══════════════════════════════════════════════════════════════════════

def unscaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    """
    Ablation version: removes division by sqrt(d_k).
    """
    scores = torch.matmul(Q, K.transpose(-2, -1))

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_w = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)

    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  TRAINING HELPERS
# ══════════════════════════════════════════════════════════════════════

def compute_token_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    pad_idx: int = PAD_IDX,
) -> float:
    """
    Token-level accuracy ignoring padding tokens.
    """
    predictions = torch.argmax(logits, dim=-1)
    mask = target.ne(pad_idx)

    correct = predictions.eq(target) & mask
    total = mask.sum().clamp_min(1)

    return (correct.sum().float() / total.float()).item()


def compute_prediction_confidence(
    logits: torch.Tensor,
    target: torch.Tensor,
    pad_idx: int = PAD_IDX,
) -> float:
    """
    Average softmax probability assigned to the correct token.
    Used for label smoothing analysis.
    """
    probs = F.softmax(logits, dim=-1)

    target_probs = probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

    mask = target.ne(pad_idx)

    if mask.sum().item() == 0:
        return 0.0

    return target_probs[mask].mean().item()


def get_qk_grad_norms(model: Transformer) -> Tuple[float, float]:
    """
    Logs gradient norms of Query and Key projection weights.
    Used for scaling factor ablation.
    """
    q_norm_sq = 0.0
    k_norm_sq = 0.0

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad_norm = param.grad.detach().norm(2).item()

        if ".W_q." in name:
            q_norm_sq += grad_norm ** 2

        if ".W_k." in name:
            k_norm_sq += grad_norm ** 2

    return math.sqrt(q_norm_sq), math.sqrt(k_norm_sq)


def _lookup_token(vocab, idx: int) -> str:
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(int(idx))

    if hasattr(vocab, "itos"):
        return vocab.itos[int(idx)]

    raise AttributeError("Vocabulary must support lookup_token(idx) or itos[idx]")


def _tokens_from_indices(indices: List[int], vocab) -> List[str]:
    tokens = []

    for idx in indices:
        token = _lookup_token(vocab, int(idx))

        if token == "<eos>":
            break

        if token in {"<unk>", "<pad>", "<sos>", "<eos>"}:
            continue

        tokens.append(token)

    return tokens


# ══════════════════════════════════════════════════════════════════════
#  EPOCH LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Autograder-facing signature preserved.
    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    for src, tgt in data_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src, PAD_IDX).to(device)
        tgt_mask = make_tgt_mask(tgt_input, PAD_IDX).to(device)

        if is_train:
            assert optimizer is not None, "optimizer must be provided during training"
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_input, src_mask, tgt_mask)

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def run_epoch_with_wandb(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    log_grad_norms: bool = False,
    max_grad_log_steps: int = 1000,
    global_step: int = 0,
) -> Tuple[float, float, float, int]:
    """
    Extended epoch loop for W&B logging.

    Returns:
        avg_loss, avg_accuracy, avg_prediction_confidence, updated_global_step
    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_confidence = 0.0
    total_batches = 0

    for src, tgt in data_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src, PAD_IDX).to(device)
        tgt_mask = make_tgt_mask(tgt_input, PAD_IDX).to(device)

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_input, src_mask, tgt_mask)

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            accuracy = compute_token_accuracy(logits, tgt_output, PAD_IDX)
            confidence = compute_prediction_confidence(logits, tgt_output, PAD_IDX)

            if is_train:
                loss.backward()

                if log_grad_norms and global_step < max_grad_log_steps:
                    q_grad_norm, k_grad_norm = get_qk_grad_norms(model)

                    wandb.log(
                        {
                            "step": global_step,
                            "grad_norm/query_weights": q_grad_norm,
                            "grad_norm/key_weights": k_grad_norm,
                        }
                    )

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                global_step += 1

        total_loss += loss.item()
        total_accuracy += accuracy
        total_confidence += confidence
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    avg_accuracy = total_accuracy / max(total_batches, 1)
    avg_confidence = total_confidence / max(total_batches, 1)

    return avg_loss, avg_accuracy, avg_confidence, global_step


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.
    """
    model.eval()

    src = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)

        ys = torch.ones(1, 1, dtype=torch.long, device=device).fill_(start_symbol)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, PAD_IDX).to(device)

            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()

            next_token_tensor = torch.ones(1, 1, dtype=torch.long, device=device).fill_(next_token)
            ys = torch.cat([ys, next_token_tensor], dim=1)

            if next_token == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU
# ══════════════════════════════════════════════════════════════════════

def _get_ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _corpus_bleu(
    predictions: List[List[str]],
    references: List[List[str]],
    max_n: int = 4,
) -> float:
    """
    Simple corpus-level BLEU implementation returning score in 0–100.
    """
    clipped_counts = [0 for _ in range(max_n)]
    total_counts = [0 for _ in range(max_n)]

    pred_len = 0
    ref_len = 0

    for pred_tokens, ref_tokens in zip(predictions, references):
        pred_len += len(pred_tokens)
        ref_len += len(ref_tokens)

        for n in range(1, max_n + 1):
            pred_ngrams = _get_ngrams(pred_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)

            total_counts[n - 1] += sum(pred_ngrams.values())

            for ngram, count in pred_ngrams.items():
                clipped_counts[n - 1] += min(count, ref_ngrams.get(ngram, 0))

    if pred_len == 0:
        return 0.0

    precisions = []

    for i in range(max_n):
        if total_counts[i] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[i] / total_counts[i])

    smooth_precisions = [
        precision if precision > 0.0 else 1e-9
        for precision in precisions
    ]

    log_precision_sum = sum(math.log(p) for p in smooth_precisions) / max_n

    if pred_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (ref_len / pred_len))

    bleu = brevity_penalty * math.exp(log_precision_sum)

    return bleu * 100.0


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Autograder-facing signature preserved.
    """
    model.eval()

    predictions = []
    references = []

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            batch_size = src.size(0)

            for i in range(batch_size):
                single_src = src[i:i + 1]
                single_tgt = tgt[i:i + 1]

                src_mask = make_src_mask(single_src, PAD_IDX).to(device)

                pred_indices = greedy_decode(
                    model=model,
                    src=single_src,
                    src_mask=src_mask,
                    max_len=max_len,
                    start_symbol=SOS_IDX,
                    end_symbol=EOS_IDX,
                    device=device,
                )

                pred_tokens = _tokens_from_indices(
                    pred_indices.squeeze(0).tolist(),
                    tgt_vocab,
                )

                ref_tokens = _tokens_from_indices(
                    single_tgt.squeeze(0).tolist(),
                    tgt_vocab,
                )

                predictions.append(pred_tokens)
                references.append(ref_tokens)

    return _corpus_bleu(predictions, references)


# ══════════════════════════════════════════════════════════════════════
#  W&B VISUALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════════

def log_attention_heatmaps(
    model: Transformer,
    src_batch: torch.Tensor,
    src_vocab,
    device: str,
    max_tokens: int = 20,
) -> None:
    """
    Logs individual attention head heatmaps from the last encoder layer.
    """
    model.eval()

    src_batch = src_batch[:1].to(device)
    src_mask = make_src_mask(src_batch, PAD_IDX).to(device)

    with torch.no_grad():
        _ = model.encode(src_batch, src_mask)

    last_encoder_layer = model.encoder.layers[-1]
    attn_weights = last_encoder_layer.self_attn.attn_weights

    if attn_weights is None:
        return

    attn_weights = attn_weights.detach().cpu()[0]

    src_indices = src_batch.detach().cpu()[0].tolist()
    src_tokens = _tokens_from_indices(src_indices, src_vocab)

    if not src_tokens:
        src_tokens = ["<empty>"]

    src_tokens = src_tokens[:max_tokens]
    seq_len = len(src_tokens)

    num_heads = attn_weights.shape[0]

    for head_idx in range(num_heads):
        heatmap_values = attn_weights[head_idx, :seq_len, :seq_len].numpy()

        table = wandb.Table(
            columns=["source_position"] + src_tokens
        )

        for row_idx, row_token in enumerate(src_tokens):
            table.add_data(row_token, *heatmap_values[row_idx].tolist())

        wandb.log(
            {
                f"attention_heatmap/head_{head_idx}": wandb.plot_table(
                    vega_spec_name="wandb/heatmap/v0",
                    data_table=table,
                    fields={
                        "x": "source_position",
                        "y": src_tokens[0],
                    },
                )
            }
        )

        wandb.log(
            {
                f"attention_matrix/head_{head_idx}": wandb.Image(
                    create_attention_figure(
                        heatmap_values,
                        src_tokens,
                        title=f"Last Encoder Layer - Head {head_idx}",
                    )
                )
            }
        )


def create_attention_figure(values, tokens, title: str):
    """
    Creates matplotlib attention heatmap figure for W&B.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(values)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))

    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)

    ax.set_xlabel("Key tokens")
    ax.set_ylabel("Query tokens")
    ax.set_title(title)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    return fig


def log_translation_examples(
    model: Transformer,
    data_loader: DataLoader,
    src_vocab,
    tgt_vocab,
    device: str,
    max_examples: int = 5,
    max_len: int = 100,
) -> None:
    """
    Logs sample German-English predictions.
    """
    model.eval()

    table = wandb.Table(columns=["source_de", "reference_en", "prediction_en"])

    count = 0

    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                single_src = src[i:i + 1]
                single_tgt = tgt[i:i + 1]

                src_mask = make_src_mask(single_src, PAD_IDX).to(device)

                pred_indices = greedy_decode(
                    model=model,
                    src=single_src,
                    src_mask=src_mask,
                    max_len=max_len,
                    start_symbol=SOS_IDX,
                    end_symbol=EOS_IDX,
                    device=device,
                )

                src_tokens = _tokens_from_indices(
                    single_src.squeeze(0).detach().cpu().tolist(),
                    src_vocab,
                )

                ref_tokens = _tokens_from_indices(
                    single_tgt.squeeze(0).detach().cpu().tolist(),
                    tgt_vocab,
                )

                pred_tokens = _tokens_from_indices(
                    pred_indices.squeeze(0).detach().cpu().tolist(),
                    tgt_vocab,
                )

                table.add_data(
                    " ".join(src_tokens),
                    " ".join(ref_tokens),
                    " ".join(pred_tokens),
                )

                count += 1

                if count >= max_examples:
                    wandb.log({"translation_examples": table})
                    return

    wandb.log({"translation_examples": table})


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimizer + scheduler state to disk.

    Autograder-facing signature preserved.
    """
    model_config = {
        "src_vocab_size": model.src_vocab_size,
        "tgt_vocab_size": model.tgt_vocab_size,
        "d_model": model.d_model,
        "N": model.N,
        "num_heads": model.num_heads,
        "d_ff": model.d_ff,
        "dropout": model.dropout_p,
    }

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": model_config,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model and optionally optimizer/scheduler state.

    Autograder-facing signature preserved.
    """
    checkpoint = torch.load(path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return int(checkpoint.get("epoch", 0))


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT SETUP
# ══════════════════════════════════════════════════════════════════════

def build_model_and_apply_ablation(config: Dict, src_vocab_size: int, tgt_vocab_size: int, device: str):
    """
    Builds model and applies chosen ablations.
    """
    if config["use_scaling"]:
        model_module.scaled_dot_product_attention = model_module.__dict__["scaled_dot_product_attention"]
    else:
        model_module.scaled_dot_product_attention = unscaled_dot_product_attention

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        load_checkpoint_by_default=False,
    ).to(device)

    if config["positional_encoding"] == "learned":
        model.positional_encoding = LearnedPositionalEncoding(
            d_model=config["d_model"],
            dropout=config["dropout"],
            max_len=config["max_len"] + 10,
        ).to(device)

    return model


def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.
    """
    args = parse_args()

    config = {
        "experiment_name": args.experiment_name,
        "d_model": args.d_model,
        "N": args.layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "warmup_steps": args.warmup_steps,
        "min_freq": args.min_freq,
        "max_len": args.max_len,
        "lr": args.lr,
        "label_smoothing": args.label_smoothing,
        "scheduler_type": args.scheduler_type,
        "use_scaling": not args.no_scaling,
        "positional_encoding": args.positional_encoding,
        "checkpoint_path": args.checkpoint_path,
        "vocab_path": args.vocab_path,
        "log_grad_norms": args.log_grad_norms,
    }

    wandb.init(
        project=args.wandb_project,
        name=args.experiment_name,
        config=config,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset = Multi30kDataset(
        split="train",
        min_freq=config["min_freq"],
        max_len=config["max_len"],
    )

    val_dataset = Multi30kDataset(
        split="validation",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        min_freq=config["min_freq"],
        max_len=config["max_len"],
    )

    test_dataset = Multi30kDataset(
        split="test",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        min_freq=config["min_freq"],
        max_len=config["max_len"],
    )

    train_dataset.save_vocabs(config["vocab_path"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=Multi30kDataset.collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=Multi30kDataset.collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=Multi30kDataset.collate_fn,
    )

    model = build_model_and_apply_ablation(
        config=config,
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    if config["scheduler_type"] == "noam":
        scheduler = NoamScheduler(
            optimizer,
            d_model=config["d_model"],
            warmup_steps=config["warmup_steps"],
        )
    elif config["scheduler_type"] == "fixed":
        scheduler = None
    else:
        raise ValueError("scheduler_type must be 'noam' or 'fixed'")

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_dataset.tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=config["label_smoothing"],
    )

    best_val_loss = float("inf")
    global_step = 0

    sample_src_batch, _ = next(iter(val_loader))

    for epoch in range(config["num_epochs"]):
        train_loss, train_acc, train_confidence, global_step = run_epoch_with_wandb(
            data_iter=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
            log_grad_norms=config["log_grad_norms"],
            max_grad_log_steps=1000,
            global_step=global_step,
        )

        val_loss, val_acc, val_confidence, global_step = run_epoch_with_wandb(
            data_iter=val_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
            log_grad_norms=False,
            global_step=global_step,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/token_accuracy": train_acc,
                "train/prediction_confidence": train_confidence,
                "val/loss": val_loss,
                "val/token_accuracy": val_acc,
                "val/prediction_confidence": val_confidence,
                "learning_rate": current_lr,
            }
        )

        print(
            f"Epoch {epoch + 1}/{config['num_epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | lr={current_lr:.8f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                path=config["checkpoint_path"],
            )

            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch

    load_checkpoint(
        path=config["checkpoint_path"],
        model=model,
        optimizer=None,
        scheduler=None,
    )

    test_bleu = evaluate_bleu(
        model=model,
        test_dataloader=test_loader,
        tgt_vocab=train_dataset.tgt_vocab,
        device=device,
        max_len=config["max_len"],
    )

    wandb.log({"test/bleu": test_bleu})
    wandb.run.summary["test_bleu"] = test_bleu

    log_translation_examples(
        model=model,
        data_loader=test_loader,
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        device=device,
        max_examples=5,
        max_len=config["max_len"],
    )

    log_attention_heatmaps(
        model=model,
        src_batch=sample_src_batch,
        src_vocab=train_dataset.src_vocab,
        device=device,
        max_tokens=20,
    )

    print(f"Test BLEU: {test_bleu:.2f}")

    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="baseline_noam_scaled_sinusoidal_ls01")
    parser.add_argument("--wandb_project", type=str, default="da6401-a3")

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1.0)

    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=100)

    parser.add_argument("--label_smoothing", type=float, default=0.1)

    parser.add_argument(
        "--scheduler_type",
        type=str,
        choices=["noam", "fixed"],
        default="noam",
    )

    parser.add_argument(
        "--positional_encoding",
        type=str,
        choices=["sinusoidal", "learned"],
        default="sinusoidal",
    )

    parser.add_argument("--no_scaling", action="store_true")
    parser.add_argument("--log_grad_norms", action="store_true")

    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--vocab_path", type=str, default="vocabs.json")

    return parser.parse_args()


if __name__ == "__main__":
    run_training_experiment()
