<h1 align="center">
  minimal identifying set → EE ⊕ GEO ⊕ derivative (WL not required)
</h1>

<p align="center">
  <img src="./banner.png" alt="btz-phase-lock banner" />
</p>

<h1 align="center">btz-phase-lock</h1>

<p align="center">
  constraint → signal > noise → structure
</p>

---

Minimal constraint-first reconstruction experiments for BTZ / AdS geometries using neural surrogates.

Core idea:  
multiple probes (EE ⊕ WL ⊕ GEO)  
→ constrain geometry  
→ signal > noise  

45° 📐

---

## 🚀 Start Here

### v20 — Probe Sweep (Minimal Identifying Set)

Probe sweep across subsets of (EE, WL, GEO) with optional derivative constraint.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thinkthoughts/btz-phase-lock/blob/main/ads4_phase_lock_v20_probe_sweep.ipynb)

---

## 🔬 v20 Result — Minimal Identifying Set

<p align="center">
  <img src="./images/v20_value_vs_shape.png" alt="v20 value vs shape fit scatter" width="48%" />
  <img src="./images/v20_metric_bar.png" alt="v20 metric error by config" width="48%" />
</p>

<p align="center">
  value fit vs shape fit across probe sets<br/>
  + metric error comparison
</p>

<p align="center">
  <img src="./images/v20_best_observables.png" width="32%" />
  <img src="./images/v20_best_metric.png" width="32%" />
  <img src="./images/v20_best_derivative.png" width="32%" />
</p>

<p align="center">
  <b>Best config:</b> EE ⊕ GEO ⊕ derivative<br/>
  <br/>
  observables ✔ &nbsp;&nbsp; metric ✔ &nbsp;&nbsp; derivative ✔  
  <br/>
  → minimal set ≠ full probe set  
  <br/>
  → WL not required for optimal reconstruction
</p>

---

## 🧠 Interpretation (v20)

- derivative constraint is **necessary** for structure  
- GEO provides **geometric anchoring**  
- WL is **redundant** under this setup  

→ minimal identifying set:

<b>EE ⊕ GEO ⊕ derivative</b>

---

## 🔬 Minimal separating probe: values → structure

<h2>🔬 v18 → v19: value fit → shape fit</h2>

<p align="center">
  <img src="./v18_v19_comparison.png" alt="v18 to v19 comparison panel" width="100%" />
</p>

<p align="center">
  v18: value fit (observables + metric)<br/>
  v19: structure fit (adds derivative constraint)<br/>
  <br/>
  bottom row contrast:<br/>
  v18 → optimization signal (loss)<br/>
  v19 → geometric signal (derivative)
</p>

---

<h2>📊 Results (v19 detail)</h2>

<p><b>Held-out test:</b><br/>
train on c = 0.00, 0.30 → predict c = 0.16
</p>

<p align="center">
  <img src="./v19_observables.png" width="32%" />
  <img src="./v19_metric.png" width="32%" />
  <img src="./v19_derivative.png" width="32%" />
</p>

<p align="center">
  <b>Observables</b> — EE ⊕ WL ⊕ GEO align<br/>
  <b>Metric</b> — geometry recovers<br/>
  <b>Derivative</b> — slope matches (new)
</p>

<p align="center">
  observables ✔ &nbsp;&nbsp; metric ✔ &nbsp;&nbsp; derivative ✔  
  <br/>
  → value fit → shape fit  
  <br/>
  → ambiguity reduced under minimal constraint
</p>

---

### v18.1 — Multi-Slice Discriminator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thinkthoughts/btz-phase-lock/blob/main/ads4_phase_lock_v18_1_multislice_discriminator.ipynb)

---

### v17 — Indistinguishable Solutions

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thinkthoughts/btz-phase-lock/blob/main/ads4_phase_lock_v17_indistinguishable.ipynb)

---

### v14–v16 — Branching → Dual Solutions → Probe Fit

- v14: branching appears  
- v15: dual/global solutions  
- v16: probe fitting improves alignment  

[Open repo notebooks](https://github.com/thinkthoughts/btz-phase-lock)

---

## 🧠 Progression

single probe → partial  

EE ⊕ WL → better  

EE ⊕ WL ⊕ GEO → stable reconstruction  

v17 → indistinguishable branches  

v18 → multi-slice reduces ambiguity  

v19 → derivative constraint matches slope (structure)  

v20 → minimal identifying set (EE ⊕ GEO ⊕ d)  

---

## 🧭 Roadmap

- v19 → derivative constraint (structure)  
- v20 → minimal identifying set (EE ⊕ GEO ⊕ d)  
- v21 → robustness (noise / seeds / slices)  

---

## ⚙️ Notes

- synthetic targets (demo-level)
- minimal MLP (tanh)
- constraint-first, not architecture-first

---

## 🌿

constraint → signal > noise  
structure → stability  

45° 📐
