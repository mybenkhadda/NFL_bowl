"""Pre-draft athletic fingerprint scorecard (Milestone 07D): DESCRIPTIVE, position-relative, uncertainty-aware.

Inputs are Combine measurements only (official results, tracked forty attempts, the frozen TCN-B representation). No
NFL value is an input of any score; Milestone 06 found no validated incremental NFL prediction, so nothing here is a
prediction of NFL performance and there is no overall athleticism score, no draft value, no risk score.

    percentiles   reference populations, oriented mid-rank percentiles, leave-one-out and fold-isolated scoring
    uncertainty   repeat-attempt measurement model, reliability-aware estimates, percentile intervals, availability
    fingerprint   TCN-B similarity: nearest athletes by position, similar-official-time athletes, repeat stability
    scorecards    section assembly, evidence confidence, Combine-side case selection, leakage guard, manifest
    figures       player scorecard and supporting figures
"""

NFL_PREDICTIVE_STATUS = ("DESCRIPTIVE ONLY — not validated as a predictor of NFL movement or performance "
                         "(Milestone 06: no incremental gain over official Combine testing was established)")
