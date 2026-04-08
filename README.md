# BTZ Phase-Lock Demo

Minimal Colab-ready demo inspired by arXiv:2604.05970  
Constraint-first reconstruction of BTZ geometry using dual neural networks.

---

## Overview

This demo implements a simplified **phase-lock setup**:

- Two tanh MLPs:
  - **L(x, ℓ)** → learns latent geometry (surface / turning point)
  - **V(r)** → learns metric function f(r)

- Alternating optimization (Adam):
  - RT (Ryu–Takayanagi) loss
  - EE (entanglement entropy)
  - WL (Wilson loops)
  - regularization + phase-lock balance

---

## Key Idea

Single constraints give partial information:

- EE → reconstructs radial structure  
- WL → complementary probe  

Combined:

EE ⊕ WL → resolves geometry

This acts as a **constraint intersection**, producing a stable reconstruction.

---

## Phase-Lock Condition

cosθ ≥ 1/√(1² + 1²) ≈ 0.707  
→ 45° alignment 📐

Interpretation:
- balanced constraints  
- stable gradients  
- bounded reconstruction error  

---

## Expected Results

After ~500 epochs:

- metric f(r) matches BTZ profile  
- reconstruction error ≲ 0.5%  
- loss stabilizes  
- gradients remain well-behaved  

---

## Run in Colab

Open directly:

https://colab.research.google.com/github/YOUR_USERNAME/btz-phase-lock/blob/main/btz_phase_lock_colab.py

---

## Outputs

The notebook generates:

- EE curve: predicted vs true  
- WL curve: predicted vs true  
- metric f(r): learned vs BTZ  
- training loss curves  
- reconstruction error  

---

## Structure

btz_phase_lock_colab.py  
README.md

---

## Notes

- This is a **toy demo**, not a full reproduction  
- Uses synthetic observables for speed and clarity  
- Designed for quick experimentation + sharing  

---

## Summary

constraint → intersection → reconstruction  

EE ⊕ WL → stable geometry  
signal remains accessible  

45° 📐

---

## License

MIT (or your choice)
