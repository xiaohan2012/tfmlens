"""Exp4 — fine-tune one decoder per capture depth on a frozen backbone.

Ties the pieces together: prior tables -> frozen forward under capture -> per-layer
decoder updates. The backbone is frozen; only the deepcopied decoders train.

Faithful to the reference recipe: the macro-batch is forwarded through the
(memory-heavy) backbone in ``micro_batch_size`` chunks, then each decoder is
updated on ``training_batch_size`` chunks of the collected test-row embeddings.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tfm_lens.adapters.base import ModelAdapter
from tfm_lens.core.capture import capture_layers
from tfm_lens.data.prior import build_prior
from tfm_lens.finetune.config import TrainConfig
from tfm_lens.utils import seed_everything


def _readout(adapter: ModelAdapter, emb, eval_pos: int):
    """Raw layer residual -> test-row, decoder-ready tensor (same as logit_lens)."""
    h = adapter.select_label_token(emb)
    h = adapter.post_norm(h)
    return h[:, eval_pos:]


def _log_step(step: int, total: int, layer_losses: list[float]) -> None:
    n = len(layer_losses)
    mid = n // 2
    print(
        f"step {step + 1:>4}/{total} | mean loss {np.nanmean(layer_losses):.3f} | "
        f"layer[0]={layer_losses[0]:.3f} layer[{mid}]={layer_losses[mid]:.3f} "
        f"layer[{n - 1}]={layer_losses[-1]:.3f}",
        flush=True,
    )


def _save(out_dir: Path, decoders: list[nn.Module], loss_per_step: list) -> None:
    for i, decoder in enumerate(decoders):
        torch.save(decoder.state_dict(), out_dir / f"decoder_layer_{i}.pth")
    (out_dir / "loss_per_step.json").write_text(json.dumps(loss_per_step))


def finetune_decoders(adapter: ModelAdapter, config: TrainConfig, prior=None) -> list[nn.Module]:
    seed_everything(config.seed)
    device = config.device
    adapter.to(device)  # co-locate the frozen backbone with decoders and inputs
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save_snapshot(out_dir)

    n_dec = adapter.n_layers + 1
    decoders = [copy.deepcopy(adapter.decoder_template()).to(device) for _ in range(n_dec)]
    optimizers = [
        torch.optim.AdamW(d.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        for d in decoders
    ]
    criterion = nn.CrossEntropyLoss()
    if prior is None:
        prior = build_prior(config)

    loss_per_step: list = []
    for step, (X, y, eval_pos) in zip(range(config.max_steps), prior, strict=False):
        # 1) forward the macro-batch through the frozen backbone in micro-batches,
        #    collecting each layer's test-row readout (kept on CPU).
        layer_embs: list[list[torch.Tensor]] = [[] for _ in range(n_dec)]
        for xb, yb in zip(
            torch.split(X, config.micro_batch_size),
            torch.split(y, config.micro_batch_size),
            strict=True,
        ):
            with torch.no_grad(), capture_layers(adapter) as cache:
                adapter.forward_frozen(xb.to(device), yb[:, :eval_pos].to(device), eval_pos)
            for i, emb in enumerate(cache):
                # Park the readout on readout_device: "cpu" offloads to keep GPU
                # peak low (portable), or the compute device to stay resident and
                # skip the round-trip when VRAM is ample (faster).
                layer_embs[i].append(_readout(adapter, emb, eval_pos).to(config.readout_device))
        embeddings = [torch.cat(chunks, dim=0) for chunks in layer_embs]
        targets = y[:, eval_pos:].long()

        # 2) one decoder update per training_batch_size chunk, per layer.
        step_losses: list[float] = []
        for emb, decoder, optimizer in zip(embeddings, decoders, optimizers, strict=True):
            losses = _update_decoder(
                decoder, optimizer, criterion, emb, targets, config, adapter.needs_transpose, device
            )
            step_losses.append(float(np.mean(losses)) if losses else float("nan"))
        loss_per_step.append(step_losses)

        if step % config.log_every == 0:
            _log_step(step, config.max_steps, step_losses)
        if (step + 1) % config.save_every == 0:
            _save(out_dir, decoders, loss_per_step)

    _save(out_dir, decoders, loss_per_step)
    return decoders


def _update_decoder(
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    emb: torch.Tensor,
    targets: torch.Tensor,
    config: TrainConfig,
    needs_transpose: bool,
    device: str,
) -> list[float]:
    losses: list[float] = []
    for b in range(0, emb.shape[0], config.training_batch_size):
        h = emb[b : b + config.training_batch_size].to(device)
        t = targets[b : b + config.training_batch_size].to(device)
        preds = decoder(h.transpose(0, 1)).transpose(0, 1) if needs_transpose else decoder(h)
        loss = criterion(preds.permute(0, 2, 1), t)
        if not torch.isfinite(loss):
            continue
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), config.grad_clip)
        optimizer.step()
        losses.append(loss.item())
    return losses
