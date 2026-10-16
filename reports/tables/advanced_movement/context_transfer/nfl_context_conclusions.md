### What the tables show (`O2.accel_sust05_mean`, yd/s²)

- **Position** carries 0.365 of all window variance — a fixed label, not an athlete difference.
- Within position, **stable athlete differences**: 0.063 [0.050, 0.072]
- Within position, **game-to-game variation of an athlete**: 0.035 [0.031, 0.040]
- Within position, **window-to-window variation inside a game**: 0.902 [0.892, 0.915]
- Marginal share of `alignment` within position: 0.051 (window level, not additive).
- Marginal share of `alignment_x_play_class` within position: 0.121 (window level, not additive).
- Marginal share of `down` within position: 0.012 (window level, not additive).
- Marginal share of `season` within position: 0.012 (window level, not additive).
- Marginal share of `starting_speed_band` within position: 0.086 (window level, not additive).
- Scale: between-athlete SD 0.287 yd/s² within position; standard error of one athlete's value 0.061 yd/s² at the median exposure (ratio 4.03).

| Definition | Athletes | Kept | Reliability [95 %] | ρ with C0 | Feasible |
|---|---|---|---|---|---|
| C0 — raw athlete aggregate (Milestone 05 statistic) | 336 | 1.00 | 0.922 [0.914, 0.956] | 1.000 | yes |
| C1 — position-standardised athlete value | 336 | 1.00 | 0.884 [0.870, 0.935] | 0.982 | yes |
| C2 — starting-speed-conditioned (position × starting-speed band) | 336 | 1.00 | 0.915 [0.897, 0.952] | 0.973 | yes |
| C3 — position × alignment conditioned | 334 | 0.99 | 0.819 [0.773, 0.884] | 0.724 | yes |
| C4 — hierarchical context-adjusted athlete effect (shrunk) | 336 | 1.00 | 0.829 [0.823, 0.887] | 0.641 | yes |

Limits: an adjustment removes what OBSERVED labels explain (alignment, play class, starting speed, season, down). Assignment, opponent and scheme are unobserved; role is partly chosen because of ability. A context-adjusted value is still observed movement — not athletic capability, and not football performance. Every C1–C4 definition is EXPLORATORY.