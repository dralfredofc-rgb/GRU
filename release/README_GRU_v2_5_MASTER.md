# GRU v2.5 MASTER — Complete Package

**The Causal Spine of Discrete Spacetime: Gravitational Wave Dispersion and LISA Falsifiability in Causal Dynamical Triangulations**

- **DOI v2.5: https://doi.org/10.5281/zenodo.21144855** (PUBLISHED)
- DOI v2.4: https://doi.org/10.5281/zenodo.20939080
- Concept DOI (always latest): https://doi.org/10.5281/zenodo.20352929
- Author: Alfredo Flores Cornejo (Zapopan, Jalisco, Mexico)
- License: CC BY 4.0

---

## OFFICIAL PARAMETERS (verified on Ubuntu server, real CDT data)

| Parameter | Value | Source |
|-----------|-------|--------|
| f_spine | **7.7912 mHz** | T/N = 20/2567, GRU_real_CDT_for_interactive.json |
| Delta_t_GRU | **128.35 s** | 1/f_spine (CDT direct) |
| Delta_t (MDR, z=1) | 277 ms | Sec 7.4.8 (LISA band, SMBHB events) |
| ds(spine) | 1.0282 +/- 0.025 | A.21, NWALKS=5000 |
| ds(full 2D) | 1.6715 +/- 0.132 | comparison script |
| ds(full 3D) | 5.0214 +/- 0.197 | real CDT N=2567 |
| gamma (MDR) | 0.0564 | 2*ds(spine)-2, verified 0.31 sigma (phenomenological hypothesis, see A.41) |
| eps_A | 4.1816% +/- 0.0006% | v2.3 universal invariant (corrected from 3.71% in v2.2) |
| lambda_corr | 2.14 +/- 0.79 slices | Screening M1, 32 CDT geometries |
| A_final | 1.46e-16 | official LISA amplitude (v2.4, Sec C.2) |
| compression_ratio | 1.626 | ds_full/ds_spine (2D) |

**IMPORTANT NOTE ON f_spine:** Some exploratory scripts in this package use
older estimates (7.877 mHz = ensemble estimate; 7.752 mHz = tautological 1/129s;
0.0323 = old 200/6200 estimate). The OFFICIAL verified value is 7.7912 mHz.
See the methodological note in Sec A.45.3 of the paper.

---

## PACKAGE STRUCTURE

```
GRU_v2_5_MASTER.zip
|
+-- README_GRU_v2_5_MASTER.md          <- this file
+-- GRU_v2_5_FINAL_CORREGIDO.html      <- full paper v2.5 (150KB)
+-- GRU_v2_5_FINAL_CORREGIDO.pdf       <- full paper PDF
+-- GRU_TAREAS_PENDIENTES_*.txt        <- task tracker
|
+-- (root scripts — LISA pipeline & core)
|   +-- GRU_F_SPINE_measure.py         <- measures f_spine from real CDT JSONs
|   +-- GRU_LISA_TIME_DELAY_TEST.py    <- Delta_t recovery test (25.98 sigma)
|   +-- GRU_LISA_TIME_DELAY_results.json
|   +-- GRU_P1_5_fermi_lat_nGRU_FIXED.py
|   +-- GRU_P1_root_invariance_test_FIXED.py  <- P1 PASS (ds=0.9602+/-0.0413)
|   +-- GRU_P1_root_invariance_v2.json
|   +-- GRU_P_ANISOTROPY_v2_5.py       <- quadrupolar anisotropy (a20=3.488e-10)
|   +-- GRU_P_PROP_FULL_v2_5_fix.py    <- exploratory (f_active pending T-scan 3D)
|   +-- GRU_P_PROP_FULL_v2_FIXED.py
|   +-- GRU_QUANTUM_FILTER.py
|   +-- GRU_QISKIT_GEOMETRIC_CHANNEL_v2.5_GRU_FIEL.py  <- rigorous QC channel
|   +-- GRU_QF_LISA_1yr_*.json         <- LISA pipeline results
|
+-- GRU_v2_5_FULL/  (31 files — Quantum Computing suite, Sec A.43)
|   +-- README_GRU_QC_v2.5_CORREGIDO.md  <- local README for QC suite
|   +-- GRU_QC_*.py + .json            <- classical numpy QC scripts (5/5 PASS)
|   +-- GRU_QISKIT_*_CORREGIDO.py      <- Qiskit baseline (historical params)
|   +-- GRU_QISKIT_*_GRU_FIEL.py       <- Qiskit rigorous (official params)
|   +-- GRU_ds_hierarchy_*             <- dimensional hierarchy (Delta_ds=4.0)
|
+-- GRU_v2_5_6_ALL_PASS/  (13 files — Challenges suite, Sec A.45)
    +-- README_GRU_scripts.md          <- local README for challenges
    +-- GRU_PPN_result.json            <- PPN PASS (delta_gamma ~1e-9 << Cassini)
    +-- GRU_SO3_result.json            <- SO(3) PASS (breaking ~1e-82 at LIGO)
    +-- GRU_challenges_pipeline_CORREGIDO.py
    +-- ward_identity_GRU_CORREGIDO.py   <- NOT in paper body (Paley-Wiener unresolved, deferred v2.6)
    +-- ringdown_echoes_GRU_CORREGIDO.py <- NOT in paper body (128s undetectable in LIGO)
    +-- *.png                          <- diagnostic figures
```

---

## WHAT IS IN v2.5 (paper sections)

| Section | Content | Status |
|---------|---------|--------|
| A.25 | V-scan robustness (V=3000/5000/8000) | included |
| A.26 | S3 x R toy 4D (ds=1.0428+/-0.0157) | included |
| A.41 | MDR from Regge Hessian (DFS 2007 framework, gamma phenomenological hypothesis, WIP derivation) | included |
| A.42 | Screening M1 phenomenological (D=4; D=3 correction -> v2.6) | included |
| A.43 | Quantum Computing Outlook (4/4 PASS, honest limitations) | included |
| A.44 | HBT topological obstruction | placeholder -> v2.6 |
| A.45 | PPN + SO(3) + Delta_t reconciliation | included |
| A.46 | Root Invariance of the Spine Protocol (P1 PASS, ds=0.9602+/-0.0413) | included |
| Sec 7.4 | LISA pipeline validation note (sigma=2678.7, synthetic) | included |
| F.2b | v2.5-specific changelog | included |

## WHAT IS NOT IN v2.5 (documented exclusions, deferred to v2.6)

- Ringdown echoes (Delta_t=128.35s undetectable in LIGO ringdown ~0.5s)
- Ward identity Paley-Wiener test (H_GRU gaussian is not PW-causal by construction;
  Ward OK=False reported honestly, not forced to PASS)
- P1 root invariance: RESOLVED in v2.5 (see A.46) — original FAIL was due to
  incorrect BFS-over-full-graph protocol, not a real issue with the spine
- sigma=2678.7 as physical detection (it is pipeline validation on synthetic data)
- Grover-LISA scenario (20 qubits, ~1000x speedup) -> v2.6
- Regge Hessian eigenvalue extraction from real CDT data -> v2.6 (blocked:
  real JSONs lack explicit simplicial structure; see project notes on TP4/TP6)
- D=3 kappa=1/(6pi) correction for Sec A.42 -> v2.6

---

## EXECUTION (Ubuntu, Python 3.10+)

```bash
# Core: measure f_spine from real CDT JSON
python3 GRU_F_SPINE_measure.py /root/GRU_real_CDT_for_interactive.json

# LISA time-delay recovery test
python3 GRU_LISA_TIME_DELAY_TEST.py

# P1 root invariance test (protocol validation)
python3 GRU_P1_root_invariance_test_FIXED.py

# QC suite (numpy only)
cd GRU_v2_5_FULL && python3 GRU_QC_CHANNEL_1QUBIT.py

# QC suite (Qiskit, rigorous version)
pip install qiskit qiskit-aer --break-system-packages
python3 GRU_QISKIT_GEOMETRIC_CHANNEL_v2.5_GRU_FIEL.py

# Challenges suite (PPN, SO3)
cd GRU_v2_5_6_ALL_PASS && python3 GRU_challenges_pipeline_CORREGIDO.py
```

---

## KEY RESULTS SUMMARY

1. **Spectral dimension universality:** ds(spine) -> 1 in CDT 2D (1.019+/-0.015),
   CDT 3D (1.039+/-0.040 or 1.0282+/-0.025 depending on protocol), toy 4D
   (1.0428+/-0.0157). Anti-triviality proven.
2. **Dimensional hierarchy (real CDT 3D):** ds(full)=5.02 -> ds(spine)=1.03,
   gap Delta_ds=4.0 confirms diffusive confinement.
3. **Root invariance (A.46):** spine is a topological invariant (S1 cycle of
   T nodes); ds=0.9602+/-0.0413 across 20 random valid representatives per
   shell, independent of representative choice (std=0.041 < 0.05, PASS).
4. **LISA falsifiability:** A_final=1.46e-16 in LISA band; Delta_t=277ms (MDR, z=1)
   and Delta_t=128.35s (CDT direct, f_spine=7.7912mHz) are distinct observables.
5. **Solar system compatibility:** PPN delta_gamma ~1e-9 (2e4 below Cassini);
   SO(3) breaking ~1e-82 at LIGO scales.
6. **QC outlook:** GRU spine defines a CPTP amplitude-damping channel
   (gamma=4.1816%, Choi rank 2); compression_ratio=1.626.

---

## NEXT STEPS (post-publication)

- GitHub Issue update to Giuseppe Clemente (@joek93) with new DOI + 3 questions
  about CDT 4D data (simplicial structure, multiple ensemble snapshots,
  generation algorithm)
- Email to Renate Loll (r.loll@science.ru.nl) with same content + arXiv
  endorsement request
- If no response: direct submission to PRD or CQG (no arXiv endorsement required)
- v2.6 roadmap: Hessian derivation of gamma, Ward identity resolution,
  f_active calibration via T-scan 3D — all blocked pending richer CDT data

---

*Generated 2026-07-02 — GRU v2.5 published on Zenodo (DOI: 10.5281/zenodo.21144855).*
