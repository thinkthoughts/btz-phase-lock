# Paper — Minimal Identifying Sets (v20)

📄 [Read the PDF](./v20_minimal_identifying_set.pdf)

---

## Summary

This paper studies constraint-first reconstruction of geometry using multiple probe observables.

Key result:

**EE ⊕ GEO ⊕ derivative → sufficient**

- lowest held-out metric error
- structure (derivative) recovered
- full probe stack is not minimal
- WL not required in this setup

---

## Context

This work was developed alongside the `btz-phase-lock` repo:

https://github.com/thinkthoughts/btz-phase-lock

Core idea:

constraint → signal > noise → structure

---

## External connection

Initial framing was influenced by:

- arXiv: https://arxiv.org/abs/2604.05970  
- Tweet: https://x.com/Precedent_Vice/status/2041833791745094091  

This raised the question:

> which observables are actually necessary to determine geometry?

---

## Project progression

- v17 → indistinguishable observables  
- v18 → multi-slice reduces ambiguity  
- v19 → derivative enforces structure  
- v20 → minimal identifying set  

---

## Figures

All figures are generated from the v20 Colab:

https://colab.research.google.com/github/thinkthoughts/btz-phase-lock/blob/main/ads4_phase_lock_v20_probe_sweep.ipynb

---

## Notes

- synthetic targets (demo-level)
- minimal MLP (tanh)
- result is setup-specific (not universal)

---

## Next

- robustness (noise / seeds)
- alternative targets
- testing whether WL becomes necessary in other regimes
