#!/usr/bin/env python3
"""GRU_QC_MDR_DEPHASING.py — MDR time-delay and dephasing across LISA band."""
import os, json, math
import numpy as np

try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz

OUTPUT_DIR=os.environ.get("GRU_OUTPUT_DIR",".")
os.makedirs(OUTPUT_DIR,exist_ok=True)

H0=70e3/3.086e22; EPl=1.956e9*1.602e-10; hP=6.626e-34; OM_M=0.30; OM_L=0.70
N_GRU=0.0564; A_FINAL=1.460e-16

def ci(z,n=N_GRU,ns=500):
    zs=np.linspace(0,z,ns)
    return trapz((1+zs)**n/np.sqrt(OM_M*(1+zs)**3+OM_L),zs)

def dt_ms(f,z,A=A_FINAL,n=N_GRU):
    return (1+n)/(2*H0)*A*(hP*f/EPl)**n*ci(z,n)*1e3

def dphi(f,z,A=A_FINAL,n=N_GRU):
    return math.pi*f*(1+n)/H0*A*(hP*f/EPl)**n*ci(z,n)

out={
    "n_GRU":N_GRU,"A_FINAL":A_FINAL,
    "delta_t_ms_3mHz_z1":dt_ms(3e-3,1.0),
    "delta_phi_3mHz_z1":dphi(3e-3,1.0),
    "detectable_dt":bool(dt_ms(3e-3,1.0)>=0.1),
    "theory_comparison":{
        "GRU_n0056":dphi(3e-3,1.0),
        "massive_gravity_n2":dphi(3e-3,1.0,A=1e-3,n=2.0),
        "SME_d5_n1":dphi(3e-3,1.0,A=1e-15,n=1.0),
        "LQG_CDT_n2":dphi(3e-3,1.0,A=1e-58,n=2.0),
    }
}
path=os.path.join(OUTPUT_DIR,"GRU_QC_MDR_DEPHASING.json")
with open(path,"w") as f: json.dump(out,f,indent=2)
print(f"Saved {path}")
print(f"  Δt={out['delta_t_ms_3mHz_z1']:.2f}ms  Δφ={out['delta_phi_3mHz_z1']:.2e}rad  detectable:{out['detectable_dt']}")
