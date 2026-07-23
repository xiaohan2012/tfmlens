---
title: Tabular Logit Lens
tags: [interpretability, mechanistic-interpretability, tabular, probing]
sources: [balef2026onelayer]
updated: 2026-05-25
---

# Tabular Logit Lens

Adaptation of the LLM "logit lens" (nostalgebraist 2020) / "tuned lens" (Belrose 2023) to tabular foundation models. Introduced by [Balef et al. 2026](balef2026onelayer.md).

## Method

For each transformer layer `l` of a frozen TFM:

1. Continue-pretrain a **fresh decoder** on synthetic datasets generated from the [TabICL](qu2025tabicl.md) prior, taking as input the hidden state at layer `l`.
2. At inference, forward the support+query through layers `1..l`, then apply decoder `l` to read out a prediction.

![Per-layer-decoder ROC-AUC across the six TFMs studied — high performance is reachable from early layers.](assets/tabular-logit-lens-early-exit.png)

## Why per-layer decoders, not the original

- TFM decoders project from the residual stream into class probabilities — they assume the final-layer representation.
- Early-layer hidden states already contain the predictive features but are not aligned with the original decoder's expected basis.
- Continued pretraining on a generic prior (TabICL's) is cheap and works across many TFMs.

## Lens vs probe

- **Probe** ([probing-classifier](probing-classifier.md)) asks *is the answer encoded?* — information-theoretic.
- **Lens** asks *would the model emit the answer?* — functional / behavioral, in the model's own output space.
- The **gap between them** is itself a diagnostic: the depth where info is encoded but not yet aligned with the original decoder.

## Use cases in Balef et al.

- **Early-exit study:** how shallow can inference be? — high AUC achievable from very early layers in all 6 TFMs studied.
- **Self-repair:** apply the lens at *every* layer after a skip intervention to see whether downstream layers compensate for the missing computation. See [self-repair](self-repair.md).
- **Prediction-ensembling gap:** the depth range where the per-layer decoder AUC has saturated but the *original* final decoder is still catching up — the residual stream still needs alignment with the original-decoder basis. Also framed as the *prediction-ensembling stage* in the paper's four-stage taxonomy ([tfm-inference-stages](tfm-inference-stages.md)).

## Limitations

- Decoder quality depends on the prior used for continued pretraining; here, TabICL priors. Models with more expressive priors (e.g. TabPFN(2.5), LimiX) may be under-served.
- Each decoder needs additional training cost (small relative to the TFM itself, but nonzero).

## Appearances in Sources

- [balef2026onelayer](balef2026onelayer.md) — introduces the method; uses it for early-exit, self-repair, and stage-identification analyses across six TFMs.
- [ferrando2024primer](ferrando2024primer.md) — surveys the LLM lens family (logit lens, tuned lens, attention lens, Patchscopes) the tabular variant adapts.

## Related Concepts

- [probing-classifier](probing-classifier.md) — information-theoretic counterpart; reads out via an external classifier.
- [self-repair](self-repair.md) — uses the lens trajectory after a skip to distinguish redundancy from active recovery.
- [tfm-inference-stages](tfm-inference-stages.md) — the per-layer-decoder vs original-decoder gap defines the *prediction-ensembling* stage.
- [looped-transformer-tfm](looped-transformer-tfm.md) — a practical alternative to per-layer decoders for any-time predictions.
