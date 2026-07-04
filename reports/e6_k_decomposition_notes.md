# E6 follow-up: what does the regime layer add over pooled adaptive K=1?

Context: the E6 K-sweep showed K=1 (pooled + adaptive rates, NO regimes)
already reaches 87.9% stress coverage on VIX-bin slices vs 88.3% for K=4 —
pooling + adaptivity carry most of the average stress repair (per-stock ACI:
84.0; per-stock pooled-off ablation: 80.2). The regime layer's incremental
value is NOT in bin-average coverage; it is in three sharper places
(numbers from the 2026-07-03 decomposition run, scripts in session log):

1. Regime transitions (the onset pit). Days-since-stress-entry profile:
   day-2 coverage K=1 75.4% vs K=4 80.2% (+4.8pp); day-3 89.5 vs 91.5.
   The regime switch applies the stress threshold IMMEDIATELY; the K=1
   tracker must climb.

2. Conditional balance of the VaR heads (alpha=0.05):
   K=1 breach rates: calm 5.26%, stress 3.62% (marginal 4.99 achieved by
   over-covering stress and under-covering calm — conditional
   miscalibration in the OPPOSITE direction to ACI).
   K=4: calm 5.00%, stress 5.71%. Balanced.
   At alpha=0.01, K=1 is better in stress (1.18 vs 1.86) — honest nuance,
   small-count regime; report both.

3. Guarantees. K>1 delivers per-regime validity by construction (Props 1-2);
   K=1's good bin-average numbers here are an empirical fact about
   persistent vol regimes + fast tracking, not a guarantee. The onset/acute
   group (VIX9D/VIX inversion, E2 round 4) is also only expressible with a
   regime layer.

Paper framing implication: decompose the mechanism honestly —
pooling >> adaptivity > regimes for average stress coverage; regimes buy
transitions, conditional balance, guarantees, and forward-looking groups.
Add K=1 as a named row in the main tables, not just the ablation.

TODO for config-driven regeneration: fold the dse-profile and VaR K-sweep
into e6_ablations.py so these numbers are script-reproducible.
