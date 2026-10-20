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
    plots      data-audit figures returning (figure, source data): audit_status_bar, valid_frames_hist, signal_panel

Shared status vocabulary and step runner: ``nflbowl.lab.status``.
Planned next (T02/T03/T07/T08): signals, spectral, models, training, preservation, latent, transfer, gates, figures.

Modules are imported explicitly; importing the package loads nothing heavy.
"""

__all__ = ["config", "data", "legacy", "optional", "plots"]
