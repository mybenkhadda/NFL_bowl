### What the tables show

- **eligible** (14): `O5.envelope_level` reliability 0.937 (n = 207); `O2.accel_sust05_mean` reliability 0.932 (n = 336); `O2.accel_sust05_p90` reliability 0.923 (n = 336); `O4.gain10_p90` reliability 0.922 (n = 336); `O3.band_acc_v0_2_mean` reliability 0.922 (n = 336); `O2.accel_sust10_p90` reliability 0.922 (n = 336); `O4.gain05_p90` reliability 0.921 (n = 336); `O1.speed_p90_mean` reliability 0.9 (n = 336); `O3.band_acc_v6_8_mean` reliability 0.884 (n = 183); `O3.band_acc_v2_4_mean` reliability 0.881 (n = 333); `O3.band_acc_v4_6_mean` reliability 0.864 (n = 301); `O4.gain_d_p90` reliability 0.856 (n = 334); `O1.speed_sust05_p90` reliability 0.844 (n = 336); `O1.speed_sust10_p90` reliability 0.84 (n = 336)
- **component** (4): `O5.band_gain_v2_4_p90` reliability 0.93 (n = 334); `O5.band_gain_v0_2_p90` reliability 0.92 (n = 336); `O5.band_gain_v6_8_p90` reliability 0.847 (n = 207); `O5.band_gain_v4_6_p90` reliability 0.809 (n = 312)
- **reference_only** (1): `O1.speed_max_p95` reliability 0.786 (n = 336)
- **exploratory** (2): `O6.decel_sust_p90` reliability 0.815 (n = 336); `O6.reaccel_gain_mean` reliability 0.8 (n = 294)

- Rule-based proposal: **FREEZE** — primary `O2.accel_sust05_mean`, secondary ['O5.envelope_level', 'O3.band_acc_v0_2_mean']. highest lower reliability bound among eligible candidates; secondaries from other families with |Spearman| < 0.9 to every chosen target.
- Recorded decision: **PENDING** — no freeze decision is recorded yet.

Reference lines (heuristics): reliability ≥ 0.7 with lower bound ≥ 0.5, ≥ 20 athletes at ≥ 50 windows in ≥ 3 position groups, smoothing rank agreement ≥ 0.9, |Spearman with log windows| ≤ 0.3, context share ≤ 0.25 (or adjusted reliability above the line).

Reliability says an outcome is repeatable for an athlete across games. It does not say the outcome is an athletic capability: role, scheme and usage are stable too, and they are part of what repeats.