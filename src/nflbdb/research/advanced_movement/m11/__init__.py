"""Milestone 11 — discovering the physics of NFL movement transfer (exploratory research programme).

AUTHORED 2026-10-14, NOT EXECUTED. Every number is produced by the user's notebook runs; nothing here holds a result.

    config          M11Config (thresholds, budgets, provenance), StudySpec registry (11A–11I), status vocabularies
    checkpoint      immutable, configuration-hashed checkpoint shards with verified manifests (Windows-safe)
    physics         speed bands, sprint model v(t), acceleration–speed envelopes, turn geometry (pure NumPy)
    segmentation    approach / braking / turning / exit / re-acceleration phases: thresholds, changepoints, a 3-state HMM
    sources         read-only access to Combine frames (all drills), officials, NFL episodes, folds and twins
    inventory       11.0: drill and NFL opportunity inventories, feasibility matrix, measurement registry, corrected gap
    combine_drills  drill-specific feature families (forty, shuttle, three-cone, transition, pass-rush, other)
    nfl_physics     NFL targets from the 09A episodes, braking→re-acceleration pairs, envelope histograms, NFL windows
    evaluation      athlete-held-out ridge ladder, paired athlete bootstrap, reliability, attenuation, controls, power
    transfer_matrix 11A · speed_state 11B · transitions 11C · development 11D · twins_transfer 11E
    representation  11F (PyTorch) · drill_selection 11G · football_actions 11H · tournament 11I
    figures, reporting, validation   figures, programmatic conclusions / report rendering, notebook-callable checks
"""
