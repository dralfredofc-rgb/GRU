#!/usr/bin/env python3
"""
GRU-LISA P1.5 FIXED: Test Fermi-LAT para n_GRU = 0.0564
=========================================================
Alfredo Flores Cornejo — GRU v2.2

CORRECCIÓN v2.2: Bug identificado y corregido (P1.5_FIX.py)
  - Bug: A_max se calculaba con n=1.0 hardcoded → A_max=2e15 (absurdo)
  - Fix: usar n=N_GRU=0.0564 en todos los cálculos de A_max
  - A_max correcto: 4.47e-14 (consistente con P1.4)

RESULTADO CENTRAL: Ventana [5.27e-20, 4.47e-14] — 5.9 órdenes
"""
import numpy as np, math
from scipy.special import gamma

try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz

H0=70e3/3.086e22; EPl=1.956e9*1.602e-10; hP=6.626e-34
Om,OL=0.30,0.70
DS_SPINE=1.0282
N_GRU=2*DS_SPINE-2   # 0.0564  — SIEMPRE usar n_GRU, nunca n=1
F_spec=gamma(N_GRU/2)/2

Z_GRB=0.903; E_PHOTON_GeV=31.0; DT_OBS_S=0.83
Z_GW=0.009; F_LIGO_HZ=100.0; DT_GW_MS=1700.0
Z_LISA=1.0; F_LISA_HZ=3e-3; DT_LISA_MS=0.1

def K_cosmo(z,n,N=800):
    zz=np.linspace(0,z,N); Eh=np.sqrt(Om*(1+zz)**3+OL)
    return trapz((1+zz)**n/Eh,zz)

def A_limit(dt_s,f_hz,z,n=N_GRU):
    """A_max desde Δt observado. SIEMPRE usar n=N_GRU."""
    Eg=hP*f_hz; K=K_cosmo(z,n)
    return dt_s/((1+n)/(2*H0)*(Eg/EPl)**n*K)

def dt_ms(f_hz,z,A,n=N_GRU):
    Eg=hP*f_hz
    return ((1+n)/(2*H0))*A*(Eg/EPl)**n*K_cosmo(z,n)*1e3

def D_C_Gpc(z):
    zz=np.linspace(0,z,800); Eh=np.sqrt(Om*(1+zz)**3+OL)
    return (3e5/70)*trapz(1/Eh,zz)/1000

if __name__=="__main__":
    print("="*65)
    print("GRU-LISA P1.5 FIXED: TEST FERMI-LAT (n_GRU=0.0564)")
    print("="*65)

    print(f"\n[DIAGNÓSTICO BUG P1.5 ORIGINAL]")
    A_bug  = A_limit(DT_OBS_S, hP*F_LIGO_HZ, Z_GW, n=1.0)
    A_fix  = A_limit(DT_GW_MS*1e-3, F_LIGO_HZ, Z_GW, n=N_GRU)
    print(f"  A_max(n=1.0, BUG):   {A_bug:.4e}  ← incorrecto")
    print(f"  A_max(n={N_GRU:.4f}):  {A_fix:.4e}  ← correcto")
    print(f"  Ratio bug/fix:       {A_bug/A_fix:.2e}")

    print(f"\n[1] n_GRU = {N_GRU:.4f}, F(d_S) = {F_spec:.6f}")

    # Límite GW170817
    A_max_GW = A_limit(DT_GW_MS*1e-3, F_LIGO_HZ, Z_GW)
    print(f"\n[2] LÍMITE GW170817 (f=100Hz, z=0.009, Δt<1700ms)")
    print(f"  A_max(GW170817) = {A_max_GW:.4e}")

    # Límite Fermi-LAT rescalado a n_GRU
    f_photon = E_PHOTON_GeV*1.602e-10/hP
    A_max_Fermi = A_limit(DT_OBS_S, f_photon, Z_GRB)
    print(f"\n[3] LÍMITE FERMI-LAT (GRB 090510, E=31GeV, z=0.903)")
    print(f"  A_max(Fermi, n_GRU) = {A_max_Fermi:.4e}")

    # Con acoplamiento diferencial f_pol=4/15 (derivado, no ansatz)
    f_pol = 4.0/15.0
    A_max_Fermi_fpol = A_max_Fermi / f_pol
    print(f"  Con f_pol=4/15 (derivado):  A_max = {A_max_Fermi_fpol:.4e}")

    # LISA mínimo
    A_min_LISA = A_limit(DT_LISA_MS*1e-3, F_LISA_HZ, Z_LISA)
    print(f"\n[4] LÍMITE LISA (f=3mHz, z=1, Δt>0.1ms)")
    print(f"  A_min(LISA) = {A_min_LISA:.4e}")

    # Ventana
    A_hi = min(A_max_GW, A_max_Fermi)
    A_hi_fpol = min(A_max_GW, A_max_Fermi_fpol)
    print(f"\n[5] VENTANA DE FALSIFICABILIDAD")
    print(f"  Sin acoplam. dif.: [{A_min_LISA:.2e}, {A_hi:.2e}]  ({math.log10(A_hi/A_min_LISA):.1f} órdenes)")
    print(f"  Con f_pol=4/15:    [{A_min_LISA:.2e}, {A_hi_fpol:.2e}]  ({math.log10(A_hi_fpol/A_min_LISA):.1f} órdenes)")

    print(f"\n[6] g_eff necesario:")
    g_max = math.sqrt(2*A_hi/F_spec)
    f_eff_max = g_max*8*math.pi
    print(f"  g_eff < {g_max:.4e}")
    print(f"  f_eff < {f_eff_max:.4e}  (vs f_spine_CDT = 7.877e-3 mHz)")
    print(f"  Supresión: {7.877e-3/f_eff_max:.2e}×")

    print(f"\n[7] Tabla Δt con A=A_hi (3mHz, varios z):")
    dc1=D_C_Gpc(1.0)
    for z in [0.1,0.5,1.0,2.0,5.0]:
        dt=dt_ms(F_LISA_HZ,z,A_hi); dc=D_C_Gpc(z)
        ok="✅" if dt>=0.1 else "❌"
        print(f"  z={z:.1f}: Δt={dt:.3e} ms  {ok}")

    import csv, os
    out = os.path.join(os.environ.get('GRU_OUTPUT_DIR','.'),
                       'GRU_P1_5_limits.csv')
    with open(out,'w') as f:
        w=csv.writer(f)
        w.writerow(['label','A','n_GRU'])
        w.writerows([['A_min_LISA',A_min_LISA,N_GRU],
                     ['A_max_GW170817',A_max_GW,N_GRU],
                     ['A_max_Fermi',A_max_Fermi,N_GRU],
                     ['A_max_Fermi_fpol',A_max_Fermi_fpol,N_GRU]])
    print(f"\nDatos: {out}")
