#!/usr/bin/env python3
"""GRU_QC_CHANNEL_1QUBIT.py — 1-qubit CPTP quantum channel inspired by GRU spine."""
import os, json
import numpy as np

OUTPUT_DIR = os.environ.get("GRU_OUTPUT_DIR", ".")
os.makedirs(OUTPUT_DIR, exist_ok=True)

I2 = np.eye(2, dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)

p_delay = 0.05; p_bulk = 0.02; theta = p_delay
U = np.cos(theta/2)*I2 - 1j*np.sin(theta/2)*Z
E0 = np.sqrt(1-p_delay-p_bulk)*U
E1 = np.sqrt(p_delay)*Z
E2 = np.sqrt(p_bulk)*(0.5*I2)

M    = E0.conj().T@E0 + E1.conj().T@E1 + E2.conj().T@E2
vals = np.linalg.eigvalsh(M)

def ch(rho): return E0@rho@E0.conj().T + E1@rho@E1.conj().T + E2@rho@E2.conj().T
def tj(a):   return {"real": a.real.tolist(), "imag": a.imag.tolist()}

rho_p = np.array([[0.5,0.5],[0.5,0.5]], dtype=complex)
rho_0 = np.array([[1.,0.],[0.,0.]], dtype=complex)

out = {
    "p_delay": p_delay, "p_bulk_loss": p_bulk, "theta": theta,
    "CPTP_eigenvalues": [float(v) for v in vals],
    "fidelity_plus": float(np.real(np.trace(rho_p @ ch(rho_p)))),
    "fidelity_zero": float(np.real(np.trace(rho_0 @ ch(rho_0)))),
    "E0": tj(E0), "E1": tj(E1), "E2": tj(E2),
    "rho_plus_out": tj(ch(rho_p)), "rho_zero_out": tj(ch(rho_0)),
}
path = os.path.join(OUTPUT_DIR, "GRU_QC_CHANNEL_1QUBIT.json")
with open(path,"w") as f: json.dump(out, f, indent=2)
print(f"Saved {path}")
print(f"  CPTP evals: {[round(v,4) for v in vals]}")
print(f"  F(|+>)={out['fidelity_plus']:.4f}  F(|0>)={out['fidelity_zero']:.4f}")
