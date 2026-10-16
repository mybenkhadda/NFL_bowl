# Phase 05 · Novelty Matrix

*Authored 2026-10-11 without online search. **No citation in this file is asserted as existing.** Novelty
status is therefore "UNVERIFIED" everywhere until the search plan in §2 is executed and each row is filled with
real, checked references (title, venue, year, URL/DOI). Do not cite anything in the submission that is not in
the verified column.*

## 1 · Matrix

Novelty is judged on the **combination** (data × method × decision), not on any method in isolation.

| ID | Contribution (hypothesis) | Nearest prior-art category to search | What would make it NOT novel | Data-specific angle (likely novel if prior art is absent) | Status |
|---|---|---|---|---|---|
| N1 | Tracking-derived Combine capacity profile (accel/decel/COD envelope from drill trajectories) | Combine-result → NFL success studies; force–velocity / sprint-profile modelling in sport science | Published use of Combine *tracking* (not results) to build athlete envelopes | Big Data Bowl 2027 is (per the brief) the first public release linking Combine tracking to game tracking for the same athletes | UNVERIFIED |
| N2 | Realized in-game **movement envelope** (speed-binned upper-quantile acceleration) per athlete, context-stratified | Acceleration–speed profiles from GPS/tracking in soccer & rugby; prior BDB entries on player speed | Same envelope estimator applied to NFL tracking with reliability-vs-exposure curves | Reliability-qualified (Phase 04 M2) envelope per rookie | UNVERIFIED |
| N3 | **Capability transfer** test: tracking capacity vs official results as predictors of the NFL envelope, athlete-grouped CV | Combine predictive-validity literature (results only) | A study already comparing tracking-derived vs official Combine predictors of in-game kinematics | Requires the 2027 data linkage | UNVERIFIED |
| N4 | **Capacity utilization** (realized / capable) as a decision metric | "Percent of max speed" workload metrics; load monitoring | Utilization defined against an *externally measured* (Combine) capacity in American football | Combine as the capacity anchor | UNVERIFIED |
| N5 | Single-agent **reachable-set counterfactual** under a swapped envelope | Reachable-region / pitch-control models; ghosting & trajectory prediction | Reachability with athlete-specific, externally calibrated dynamic limits | Envelope from Combine, calibrated on held-out athletes | UNVERIFIED |
| N6 | Offline policy evaluation over vendor action labels with explicit **non-identifiability audit** | Offline RL / OPE in sports; "ghosting" and RL play evaluation | — (an honest negative feasibility result is a methodological contribution, not a novelty claim) | Overlap/ESS diagnostics on a cohort-only dataset | UNVERIFIED |
| N7 | Rookie-vs-rookie co-tracked matchup analysis | Receiver–defender separation models, BDB coverage entries | — | Cohort-limited; novelty low unless paired with N1 envelopes | UNVERIFIED |

## 2 · Literature search plan (to execute when online research is allowed)

**Sources:** Google Scholar, Semantic Scholar, arXiv (cs.LG, stat.AP), MIT Sloan Sports Analytics Conference
papers, *Journal of Quantitative Analysis in Sports*, *Journal of Sports Analytics*, *Journal of Strength and
Conditioning Research*, *International Journal of Sports Physiology and Performance*, Kaggle NFL Big Data Bowl
winner/finalist write-ups (2019–2026) and the NFL Football Operations BDB pages.

**Queries (each row of §1):**
- N1/N3: "NFL combine predictive validity", "combine tracking data", "40-yard dash split tracking", "sprint force-velocity profile American football", "combine performance NFL success regression".
- N2: "acceleration speed profile in-situ", "acceleration-speed relationship match tracking", "player tracking maximal acceleration NFL", "Big Data Bowl speed acceleration".
- N4: "percentage of maximal speed match", "relative intensity workload tracking football".
- N5: "reachable set player tracking", "pitch control model", "time to intercept football tracking", "ghosting trajectory prediction defense".
- N6: "offline reinforcement learning sports", "off-policy evaluation football play calling", "counterfactual play evaluation tracking".
- N7: "receiver separation model tracking", "coverage tracking matchup Big Data Bowl".

**Protocol:** for each query record (date, source, query, top-20 screened, relevant hits). A row is
**NOVEL-PLAUSIBLE** only if no screened hit implements the "what would make it NOT novel" condition; otherwise
**PRIOR-ART** with the reference(s) stored in `reports/tables/phase_05/literature_screening.csv`
(columns: novelty_id, source, query, date, title, venue, year, url_or_doi, relevance, verdict, verified_by).

## 3 · Positioning statement (draft — valid only after §2)

"We use the first public linkage of Combine and game tracking for the same athletes to test whether
drill-measured movement capacity transfers to in-game movement, and turn the gap into a utilization metric for
scouting and development." — subject to N1–N4 surviving the search.
