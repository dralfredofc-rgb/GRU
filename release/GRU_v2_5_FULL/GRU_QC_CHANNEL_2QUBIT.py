#!/usr/bin/env python3
"""GRU_QC_CHANNEL_2QUBIT.py — 2-qubit CPTP channel with GRU geometric noise."""
import os, json
import numpy as np

OUTPUT_DIR = os.environ.get("GRU_OUTPUT_DIR", ".")
os.makedirs(OUTPUT_DIR, exist_ok=True)

I2 = np.eye(2,dtype=complex); Z = np.array([[1,0],[0,-1]],dtype=complex)
p_delay=0.05; p_bulk=0.02; theta=p_delay
U = np.cos(theta/2)*I2 - 1j*np.sin(theta/2)*Z
I4=np.kron(I2,I2); U2=np.kron(U,U); ZI=np.kron(Z,I2); IZ=np.kron(I2,Z)
E0=np.sqrt(1-p_delay-p_bulk)*U2
E1=np.sqrt(p_delay/2)*ZI
E2=np.sqrt(p_delay/2)*IZ
E3=np.sqrt(p_bulk)*(0.25*I4)

def ch2(r): return E0@r@E0.conj().T+E1@r@E1.conj().T+E2@r@E2.conj().T+E3@r@E3.conj().T
def tj(a): return {"real":a.real.tolist(),"imag":a.imag.tolist()}

def concurrence(rho4):
    Y=np.array([[0,-1j],[1j,0]],dtype=complex)
    YY=np.kron(Y,Y)
    R=rho4@(YY@np.conj(rho4)@YY)
    lam=np.sort(np.sqrt(np.maximum(np.real(np.linalg.eigvals(R)),0)))[::-1]
    return max(0., lam[0]-lam[1]-lam[2]-lam[3])

z0=np.array([1,0],dtype=complex); z1=np.array([0,1],dtype=complex)
ket00=np.kron(z0,z0); ket11=np.kron(z1,z1)
ket_b=(ket00+ket11)/np.sqrt(2)
rho00=np.outer(ket00,ket00.conj()); rho_b=np.outer(ket_b,ket_b.conj())
rho00_out=ch2(rho00); rho_b_out=ch2(rho_b)

out={
    "fidelity_00":float(np.real(np.trace(rho00@rho00_out))),
    "fidelity_bell":float(np.real(np.trace(rho_b@rho_b_out))),
    "concurrence_in":float(concurrence(rho_b)),
    "concurrence_out":float(concurrence(rho_b_out)),
    "delta_C":float(concurrence(rho_b)-concurrence(rho_b_out)),
    "rho_bell_out":tj(rho_b_out),
}
path=os.path.join(OUTPUT_DIR,"GRU_QC_CHANNEL_2QUBIT.json")
with open(path,"w") as f: json.dump(out,f,indent=2)
print(f"Saved {path}")
print(f"  F(Bell)={out['fidelity_bell']:.4f}  C_in={out['concurrence_in']:.4f}  C_out={out['concurrence_out']:.4f}  ΔC={out['delta_C']:.4f}")
