"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

from collections import Counter
from typing import Optional, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token.
        smoothing  (float): Smoothing factor ε.
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
            logits : shape [batch * tgt_len, vocab_size]
            target : shape [batch * tgt_len]

        Returns:
            Scalar loss.
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

            target_unsqueezed = target.unsqueeze(1)
            true_dist.scatter_(1, target_unsqueezed, self.confidence)

            pad_mask = target.eq(self.pad_idx)
            true_dist[pad_mask] = 0.0

        loss = -(true_dist * log_probs).sum(dim=-1)

        non_pad_tokens = target.ne(self.pad_idx).sum().clamp_min(1)
        return loss.sum() / non_pad_tokens


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP
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


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING
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
#   BLEU HELPERS
# ══════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.
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
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.
    """
    import wandb

    config = {
        "d_model": 256,
        "N": 3,
        "num_heads": 8,
        "d_ff": 512,
        "dropout": 0.1,
        "batch_size": 64,
        "num_epochs": 10,
        "warmup_steps": 4000,
        "min_freq": 2,
        "max_len": 100,
        "lr": 1.0,
        "label_smoothing": 0.1,
        "checkpoint_path": "checkpoint.pt",
        "vocab_path": "vocabs.json",
    }

    wandb.init(project="da6401-a3", config=config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = Transformer(
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    scheduler = NoamScheduler(
        optimizer,
        d_model=config["d_model"],
        warmup_steps=config["warmup_steps"],
    )

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_dataset.tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=config["label_smoothing"],
    )

    best_val_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        train_loss = run_epoch(
            data_iter=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
        )

        val_loss = run_epoch(
            data_iter=val_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
            }
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

    wandb.log({"test_bleu": test_bleu})
    print(f"Test BLEU: {test_bleu:.2f}")


if __name__ == "__main__":
    run_training_experiment()