"""``nflbowl.lab.m15`` — Milestone 15: spectral & temporal representation science.

Notebook: ``notebooks/nflbowl/M15_SPECTRAL_TEMPORAL_REPRESENTATION_LAB_COLAB.ipynb``.
Specification: ``docs/research/M15_M18_*.md`` (experiment IDs E15.00–E15.E3, gate G15). Authored, not executed.

    config        M15Config / resolve / M15Run (layout, read-only guards, fingerprint, config snapshots, status, hand-off)   [T01]
    optional      lazy optional imports (missing package → BLOCKED_MISSING_INPUT for the dependent step only)              [T01]
    data          E15.00 sequence-store adapter: audit, fold assignments, sequence index, counts, game key                  [T01]
    legacy        provenance audit of the exploratory STFT / LSTM artifacts, assert_fold_safe, adopt                         [T01]
    signals       E15.A1/A2 time-domain features, validity, length-matched crops, matched domain contrasts                    [T02]
    spectral      E15.A3/A4 Welch / STFT / CWT, resolution-artifact test, reliability, sensitivity, measurement status       [T03]
    models        E15.B1 PCA, STFT-CNN-AE, BiLSTM / BiGRU-AE, TCN-AE (m11 ConvEncoder), masked TCN; equal-budget widths     [T07]
    training      fold-safe, resumable, budget-matched training; embedding export with roles and provenance                 [T07]
    preservation  E15.C1/C2 valid-frame physical errors, event preservation (M14 detector), band errors, robustness          [T08]
    latent        E15.D1–D3 geometry, probes, per-dimension ICC, repeat AUC, seed stability, interpretation, projections    [T08]
    transfer      E15.E1–E3 retrieval pools / arms, Procrustes, InfoNCE heads, position-only chance, damage check           [T08]
    gates         G15 decisions from the saved tables                                                                        [T08]
    pipeline      notebook orchestration of B–E and G15; plots: every figure returns (figure, source data)

Shared status vocabulary and step runner: ``nflbowl.lab.status``. Modules are imported explicitly.
"""

__all__ = ["config", "data", "gates", "latent", "legacy", "models", "optional", "pipeline", "plots", "preservation", "signals", "spectral", "training",
           "transfer"]
