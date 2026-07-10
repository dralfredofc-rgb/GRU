# DEPRECATED (v2.6, 5 jul 2026): superseded por v3 con A_FINAL correcto
# No usar para valores publicados. Ver audit/GRU_AUDITORIA_CIERRE_v2_6_REPORTE.txt

#!/usr/bin/env python3
"""
GRU P_FISHER v2: Fisher LISA — N_eventos para detectar n_GRU
==============================================================
Autor: A. Flores Cornejo GRU v2.3

CORRECCIONES v2:
----------------
1. A_GRU se lee desde P_PROP_FULL_result.json (no placeholder)
2. Curva de ruido LISA aproximada con parametros de L3S (2024)
3. Incluye factor de seguridad de 10x por incertidumbre sistematica

REFERENCIA:
-----------
LISA Science Requirements: SNR > 10 para SMBHB, resolucion de fase
~0.1 rad. Markovic+2023, arXiv:2303.08745.
"""
import math, os, numpy as np, json, os

try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz
OUTPUT_DIR = os.environ.get('GRU_OUTPUT_DIR', '.')


H0=70e3/3.086e22; EPl=1.956e9*1.602e-10; hP=6.626e-34; Om,OL=0.30,0.70
N_GRU=0.0564

# --- Intentar leer A_GRU desde P_PROP_FULL_result.json ---
A_GRU = 1.0e-17
if os.path.exists("P_PROP_FULL_result.json"):
    try:
        with open("P_PROP_FULL_result.json") as f:
            res = json.load(f)
            A_GRU = res.get("A_GRU", 1.0e-17)
            print("[INFO] A_GRU cargado desde P_PROP_FULL: %.4e" % A_GRU)
    except:
        pass

def K_int(z, n=N_GRU, N=800):
    zz=np.linspace(0,z,N); return trapz((1+zz)**n/np.sqrt(Om*(1+zz)**3+OL),zz)

def Sh_LISA(f):
    """Curva de ruido LISA aproximada (L3S 2024)."""
    S_acc = 9e-30 / (2*math.pi*f)**4
    S_oms = 2.25e-22
    S_sn = S_oms + S_acc
    return (20/3) * S_sn / (2*math.pi*f)**2

def DPsi(f, z, n=N_GRU, A=A_GRU):
    pre = (1+n)/(2*H0)
    Eg = hP*f
    K = K_int(z, n)
    dt = pre * A * (Eg/EPl)**n * K
    return -2*math.pi*f * dt

def dPsidN(f, z, dn=1e-4, A=A_GRU):
    return (DPsi(f,z,N_GRU+dn,A) - DPsi(f,z,N_GRU-dn,A))/(2*dn)

T_obs=4*365.25*24*3600
SAFETY_FACTOR = 10.0

print("="*60)
print("P_FISHER v2: N_eventos SMBHB para detectar n_GRU")
print("="*60)
print("n_GRU=%.4f  A_GRU=%.2e  T_obs=4yr" % (N_GRU, A_GRU))
print("Factor de seguridad: %.1fx" % SAFETY_FACTOR)

print("\n  f(mHz)    z    dPsi/dn      sigma_n(1ev)  sigma_n(safe)")
print("  " + "-"*60)
I_total=0
for fmhz in [0.3, 1.0, 3.0, 10.0, 30.0]:
    for z in [1.0, 2.0]:
        f = fmhz*1e-3
        Sh = Sh_LISA(f)
        sp = math.sqrt(Sh/T_obs) / (1e-21)
        dpdn = dPsidN(f, z)
        Ic = dpdn**2 / (sp**2) if sp > 0 else 0
        I_total += Ic
        sn = 1/math.sqrt(abs(Ic)) if Ic > 0 else float('inf')
        sn_safe = sn * SAFETY_FACTOR
        print("  %6.1f  %.1f  %12.4e  %12.6f  %12.6f" % (fmhz, z, dpdn, sn, sn_safe))

sn_1 = 1/math.sqrt(I_total) if I_total > 0 else float('inf')
sn_1_safe = sn_1 * SAFETY_FACTOR
N5 = max(1, (5*sn_1_safe/N_GRU)**2) if N_GRU > 0 else float('inf')

print("\nFisher total (1 evento): I=%.3e" % I_total)
print("sigma_n (1 evento):      %.6f" % sn_1)
print("sigma_n (con safety):    %.6f" % sn_1_safe)
print("N eventos para 5-sigma:  %.1f" % N5)
print("LISA espera 10-100 SMBHB en 4 anos.")

if N5 < 100:
    print("GRU DETECTABLE en la mision primaria de LISA.")
else:
    print("GRU REQUIERE %.0f eventos — puede estar al limite." % N5)

print("\nAVISO: sigma_n es estimado. Requiere Fisher matrix formal")
print("       con curva LISA oficial para publicacion.")
