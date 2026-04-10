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

### v19 — Minimal Separating Probe
Adds derivative constraint to match **slope (structure)**, not just values.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thinkthoughts/btz-phase-lock/blob/main/ads4_phase_lock_v19_minimal_separating_probe.ipynb)

---

### v18.1 — Multi-Slice Discriminator
Train on multiple slices → test held-out interpolation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thinkthoughts/btz-phase-lock/blob/main/ads4_phase_lock_v18_1_multislice_discriminator.ipynb)

---

### v17 — Indistinguishable Solutions
Different solutions produce nearly identical observables.

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

---

## 🔬 Key Result

v19 shows:

- observables match ✔  
- metric reconstructs ✔  
- derivative matches (new) ✔  

→ value fit → shape fit  
→ “looks right” ≠ “is right” → now constrained  

---

## 📊 Example (v19)

- held-out slice: c = 0.16  
- train: c = 0.00, 0.30  

results:

- EE / WL / GEO align  
- metric recovers  
- derivative aligns  

---

## 🧭 Roadmap

- v19 → minimal separating probe  
- v20 → probe sweep (which constraint matters most?)  
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
