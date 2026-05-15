# Web Fingerprinting Defences — Experiments

Companion code for the Data and Network Security report on:
- **Simulacrum** (Karami et al., USENIX Security 2022) — extension fingerprinting defence
- **QCSD** (Smith et al., USENIX Security 2022) — website fingerprinting defence via QUIC

Two independent experiments were implemented to validate claims from the papers.

---

## Files

| File | What it does |
|---|---|
| `front_schedule.py` | Implements the FRONT chaff-schedule generator from scratch |
| `kfp_classifier.py` | Reproduces the k-Fingerprinting (k-FP) attack classifier |

---

## Experiment 1 — FRONT Schedule Generator (`front_schedule.py`)

Implements the FRONT defence schedule from **Appendix A of Smith et al.**

FRONT adds fake (chaff) packets to disguise real traffic. Packet timestamps are drawn
from a Rayleigh distribution:

```
f(t; w) = (t / w²) · exp(−t² / 2w²),  t ≥ 0
```

Parameters used (from Section 4.2 of the paper):
- `Nc = Ns = 1000` — max chaff packets per direction
- `Wmin = 0.5 s`, `Wmax = 7.0 s` — uniform range for scale parameter
- Packet size = 1200 bytes

**Output:** `front_schedule.png` — 5 independently sampled schedules showing
outgoing chaff (blue, above axis) and incoming chaff (green, below axis).

### Run

```bash
pip install numpy matplotlib
python front_schedule.py
```

---

## Experiment 2 — k-FP Classifier (`kfp_classifier.py`)

Reproduces the **k-Fingerprinting attack** (Hayes & Danezis, USENIX Security 2016),
the same baseline used in the QCSD evaluation.

k-FP is a Random Forest that extracts traffic features (total bytes, packet counts,
burst count, inter-arrival time statistics, initial packet sizes) from a network
trace and classifies which website was visited.

**Dataset:** Synthetic open-world dataset with:
- 15 monitored sites, 60 traces each (site-specific burst patterns and packet sizes)
- 400 unmonitored traces with random properties

**Setup:** 200-tree Random Forest, sqrt feature sampling, 80/20 stratified split.

**Result:** Macro F1 = 0.994 — consistent with the >94% accuracy reported in the
literature, confirming the severity of the attack before any defence is applied.

**Output:** `kfp_results.png` — confusion matrix + per-class precision/recall.

### Run

```bash
pip install numpy matplotlib scikit-learn
python kfp_classifier.py
```

---

## Requirements

```
numpy
matplotlib
scikit-learn
```

Install all at once:

```bash
pip install numpy matplotlib scikit-learn
```

---

## References

- Smith et al., *QCSD: A QUIC Client-Side Website-Fingerprinting Defence Framework*, USENIX Security 2022
- Karami et al., *Unleash the Simulacrum*, USENIX Security 2022
- Hayes & Danezis, *k-fingerprinting*, USENIX Security 2016
- Gong & Wang, *Zero-delay Lightweight Defenses Against Website Fingerprinting (FRONT)*, USENIX Security 2020
