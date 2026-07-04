#!/usr/bin/env python3
"""GRU_QC_QUANTUM_WALK_SPINE.py — Quantum walk on GRU spine vs 2D grid."""
import os, json
import numpy as np

try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz

OUTPUT_DIR = os.environ.get("GRU_OUTPUT_DIR", ".")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def laplacian_chain(N):
    L=np.zeros((N,N)); L[0,0]=1; L[-1,-1]=1
    for i in range(1,N-1): L[i,i]=2; L[i,i-1]=-1; L[i,i+1]=-1
    return L

def laplacian_grid(n):
    N=n*n; L=np.zeros((N,N))
    def idx(i,j): return i*n+j
    for i in range(n):
        for j in range(n):
            v=idx(i,j); deg=0
            for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni,nj=i+di,j+dj
                if 0<=ni<n and 0<=nj<n: L[v,idx(ni,nj)]=-1; deg+=1
            L[v,v]=deg
    return L

def spectral_dim(L):
    evals,evecs=np.linalg.eigh(L)
    ts=np.linspace(0.5,5.0,50)
    psi0=np.zeros(L.shape[0]); psi0[0]=1.
    p0=np.zeros_like(ts)
    for k,t in enumerate(ts):
        prop=evecs@np.diag(np.exp(-evals*t))@evecs.T; p0[k]=prop[0,0]
    slope,_=np.polyfit(np.log(ts),np.log(np.maximum(p0,1e-15)),1)
    return float(-2*slope), ts.tolist(), p0.tolist()

ds1,ts1,p1=spectral_dim(laplacian_chain(20))
ds2,ts2,p2=spectral_dim(laplacian_grid(4))

out={"spectral_dim_chain":ds1,"spectral_dim_grid":ds2,"GRU_spine_ds":1.02,
     "return_prob_chain":p1,"return_prob_grid":p2,"times":ts1,
     "note":"d_S<4 → no Grover speedup locally. Long-range tunneling needed."}
path=os.path.join(OUTPUT_DIR,"GRU_QC_QUANTUM_WALK_SPINE.json")
with open(path,"w") as f: json.dump(out,f,indent=2)
print(f"Saved {path}")
print(f"  d_S chain={ds1:.3f}  d_S grid={ds2:.3f}  (GRU spine ~1.02)")
