# 12F — Physics-informed shared movement encoder

**Status:** authored 2026-10-15 · execution pending (may be deferred; needs PyTorch). Module `m12/shared_movement_encoder.py` (reuses `m11/representation.py`); notebook `12f_physics_informed_shared_movement_encoder.ipynb`.

## Input
One immutable memory-mapped sequence store: every Combine attempt (≤ 6.4 s) and a deterministic capped sample of NFL episode windows (4.8 s; ≤ 40 per athlete × demand, ≤ 200,000) from one resumable tracking scan. Six channels in fixed physical units; masks for invalid / padded frames (padded inputs are zeroed before every encoder call).

## Objectives
L = λ_mask L_masked + λ_phys L_physics + λ_temp L_temporal + λ_align L_cross-domain. Physics targets come from each sequence itself (max speed, speed gain, mean a_tan, max |a_n|, |heading change| only with valid heading, duration). Alignment variants: none · matched speed state (supervised contrastive across domains) · adversarial (gradient reversal) · MMD · movement category.

## Families
F0 engineered · F1 TCN-B (Milestone 06, Combine forty only) · F2 CNN + physics · F3 masked AE · **F4 physics-informed contrastive + matched-state alignment (primary)** · F5 JEPA-style + physics · F6 small transformer · ablations F4_nophys, F4_noalign, F4_adv, F4_mmd, F4_category · control F4_shuffled_phys · random frozen encoder.

## Regimes
Transductive (all sequences; label-free diagnostics) and inductive (per outer fold without the held-out athletes' sequences of either domain; the only regime in NFL prediction) — separate tables. Immutable epoch shards, wall-clock cap, early stopping on held-out athletes, AMP / CPU fallback, embedding caches with signatures.

<!-- M12:RESULTS:START -->
_execution pending — written by 12F._
<!-- M12:RESULTS:END -->
