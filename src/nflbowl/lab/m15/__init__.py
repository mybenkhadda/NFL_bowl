"""``nflbowl.lab.m15`` — Milestone 15: spectral & temporal representation science.

Notebook: ``notebooks/nflbowl/M15_SPECTRAL_TEMPORAL_REPRESENTATION_LAB_COLAB.ipynb``.
Specification: ``docs/research/M15_M18_*.md`` (experiment IDs E15.00–E15.E3, gate G15).

Implemented (task T01; authored, not executed):

    config     M15Config / resolve / M15Run (layout, read-only guards, fingerprint, config snapshots, status, hand-off)
    optional   lazy optional imports (missing package → BLOCKED_MISSING_INPUT for the dependent step only)
    data       E15.00 M12 sequence-store adapter: open_m15_bundle, audit_bundle, fold_assignments, sequence_index,
               counts_tables, game_key, prepare_m15_data, physical_rows, athlete_weighted_rows
    legacy     provenance audit of the exploratory STFT / LSTM artifacts (verified_fold_safe ·
               verified_exploratory_only · unknown_provenance), assert_fold_safe, adopt (copy + sidecar)
    plots      figures returning (figure, source data): audit_status_bar, valid_frames_hist, signal_panel (T01);
               contrast_waterfall, kind_heatmap (T02); centroid_vs_length, psd_by_domain, spectrogram_gallery, icc_forest (T03)

Task T02 (E15.A1–A2):
    signals    per-sequence time-domain features on valid frames, validity tables, length-matched Combine crops,
               four-step matched domain contrast (raw → athlete-weighted → length-matched → CEM), kind profiles; run_a1/run_a2

Task T03 (E15.A3–A4):
    spectral   NumPy Welch / STFT / Morlet CWT, spectral features (DC excluded), ramp and white-noise surrogates,
               resolution-artifact test, length-matched domain contrast, ICC reliability, NFL split-half, M14 perturbation
               sensitivity, measurement status (M14 rule); run_a3/run_a4

Shared status vocabulary and step runner: ``nflbowl.lab.status``.
Planned next (T07/T08): models, training, preservation, latent, transfer, gates.

Modules are imported explicitly; importing the package loads nothing heavy.
"""

__all__ = ["config", "data", "legacy", "optional", "plots", "signals", "spectral"]
