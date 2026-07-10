#!/usr/bin/env python3
"""GRU_QC_CHOI_ANALYSIS.py — Choi matrix rank for 1q and 2q GRU channels."""
import os, json
import numpy as np

OUTPUT_DIR=os.environ.get("GRU_OUTPUT_DIR",".")
os.makedirs(OUTPUT_DIR,exist_ok=True)

I2=np.eye(2,dtype=complex); Z=np.array([[1,0],[0,-1]],dtype=complex)
p=0.05; pb=0.02; t=p
U=np.cos(t/2)*I2-1j*np.sin(t/2)*Z
E0_1=np.sqrt(1-p-pb)*U; E1_1=np.sqrt(p)*Z; E2_1=np.sqrt(pb)*(0.5*I2)
def ch1(r): return E0_1@r@E0_1.conj().T+E1_1@r@E1_1.conj().T+E2_1@r@E2_1.conj().T

I4=np.kron(I2,I2); U2=np.kron(U,U); ZI=np.kron(Z,I2); IZ=np.kron(I2,Z)
E0_2=np.sqrt(1-p-pb)*U2; E1_2=np.sqrt(p/2)*ZI; E2_2=np.sqrt(p/2)*IZ; E3_2=np.sqrt(pb)*(0.25*I4)
def ch2(r): return E0_2@r@E0_2.conj().T+E1_2@r@E1_2.conj().T+E2_2@r@E2_2.conj().T+E3_2@r@E3_2.conj().T

def choi(ch,dim):
    C=np.zeros((dim*dim,dim*dim),dtype=complex)
    for i in range(dim):
        for j in range(dim):
            ei=np.zeros(dim); ei[i]=1.; ej=np.zeros(dim); ej[j]=1.
            C+=np.kron(np.outer(ei,ej), ch(np.outer(ei,ej)))
    return C/dim

C1=choi(ch1,2); C2=choi(ch2,4)
ev1=np.linalg.eigvalsh(C1).real; ev2=np.linalg.eigvalsh(C2).real
r1=int(np.sum(np.abs(ev1)>1e-8)); r2=int(np.sum(np.abs(ev2)>1e-8))

out={"choi_rank_1q":r1,"choi_rank_2q":r2,
     "evals_1q":ev1.tolist(),"evals_2q_nz":[float(v) for v in ev2 if abs(v)>1e-8],
     "interpretation":"Low Choi rank = structured geometric noise, not arbitrary."}
path=os.path.join(OUTPUT_DIR,"GRU_QC_CHOI_ANALYSIS.json")
with open(path,"w") as f: json.dump(out,f,indent=2)
print(f"Saved {path}")
print(f"  Choi rank 1q={r1}  2q={r2}")
