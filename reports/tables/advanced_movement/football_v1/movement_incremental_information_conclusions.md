### 08B.4 — incremental Combine movement information

| Candidate | Metric | Raw [95 %] | R² official | After official tests [95 %] | Status |
|---|---|---|---|---|---|
| tracked top speed (p95) | ICC(1) within position | 0.95 [0.94, 0.96] | 0.97 | 0.68 [0.62, 0.75] | repeatable_beyond_official |
| speed development 4 → 8 yd/s | ICC(1) within position | 0.82 [0.77, 0.86] | 0.88 | 0.59 [0.49, 0.67] | repeatable_beyond_official |
| acceleration over 1 s from 4 yd/s | ICC(1) within position | 0.67 [0.57, 0.75] | 0.82 | 0.47 [0.34, 0.59] | unstable |
| speed gain over 10 yd from 4 yd/s | ICC(1) within position | 0.81 [0.75, 0.86] | 0.92 | 0.49 [0.38, 0.59] | weak_residual_signal |
| FPCA speed-curve scores (K = 2) | repeat-attempt retrieval AUC within position | 0.78 [0.75, 0.80] | 0.95 | 0.74 [0.72, 0.76] | repeatable_beyond_official |
| TCN-B movement fingerprint (8-d) | repeat-attempt retrieval AUC within position | 0.97 [0.97, 0.98] | 0.10 | 0.98 [0.98, 0.98] | repeatable_beyond_official |

- `F2.forty.p95_speed`: 49.46 % of 833 matched stopwatch pairs differ by more than their measurement error
- `F6.forty.accel_1s_from_v4`: 19.81 % of 833 matched stopwatch pairs differ by more than their measurement error
- `F6.forty.speed_gain_10yd_from_v4`: 25.09 % of 833 matched stopwatch pairs differ by more than their measurement error
- `F6.forty.time_v4_to_v8_s`: 21.85 % of 833 matched stopwatch pairs differ by more than their measurement error

Reference lines (heuristics): scalar raw ICC ≥ 0.70, residual ICC ≥ 0.50 with a positive lower bound; embedding raw AUC ≥ 0.75, residual AUC ≥ 0.65 with a lower bound above 0.5. Repeatable beyond the stopwatch means NEW MEASURED INFORMATION about the athlete's run — it says nothing about NFL performance.