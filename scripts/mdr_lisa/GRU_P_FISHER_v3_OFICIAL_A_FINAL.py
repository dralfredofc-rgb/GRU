#!/usr/bin/env python3
"""
GRU_P_FISHER_v3_OFICIAL_A_FINAL.py
==================================================================
Reemplaza a GRU_P_FISHER_v2_FIXED.py como fuente oficial de:
  - sigma_n (1 evento) y sigma_n_safe, calculados con A_final
    (post-screening), NO con A_GRU pre-screening.
  - N5 (eventos para 5-sigma) con A_final.
  - Separaciones en sigma respecto a n=2 (masiva/LQG) y n=1 (SME d=5),
    usadas en el paper (Sec. 7.4.8) y que hasta ahora seguian usando
    el sigma_n viejo (0.0027, calculado con A pre-screening).

Este script NO lee ningun JSON externo: usa A_final=1.46e-16 de forma
explicita y documentada, para evitar el error de origen (leer
"A_GRU" de P_PROP_FULL_result.json, que es la amplitud pre-screening).

Uso: python3 GRU_P_FISHER_v3_OFICIAL_A_FINAL.py
Salida: imprime en pantalla y escribe
        GRU_P_FISHER_v3_OFICIAL_A_FINAL_result.json
==================================================================
"""
import math, json
import numpy as np

try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz

# ------------------------------------------------------------------
# CONSTANTES FISICAS (identicas a GRU_P_FISHER_v2_FIXED.py)
# ------------------------------------------------------------------
H0 = 70e3 / 3.086e22
EPl = 1.956e9 * 1.602e-10
hP = 6.626e-34
Om, OL = 0.30, 0.70

N_GRU = 0.0564                    # exponente MDR de GRU (A.41)
A_FINAL = 1.46e-16                 # AMPLITUD OFICIAL POST-SCREENING
                                    # (NO usar A_GRU=1.9148e-08 pre-screening)
SAFETY_FACTOR = 10.0
T_obs = 4 * 365.25 * 24 * 3600

# Exponentes de teorias competidoras, para las separaciones del §7.4.8
N_ALT_MASSIVE_GRAVITY_LQG = 2.0
N_ALT_SME_D5 = 1.0


def K_int(z, n=N_GRU, N=800):
    zz = np.linspace(0, z, N)
    return trapz((1 + zz) ** n / np.sqrt(Om * (1 + zz) ** 3 + OL), zz)


def Sh_LISA(f):
    """Curva de ruido LISA aproximada (L3S 2024)."""
    S_acc = 9e-30 / (2 * math.pi * f) ** 4
    S_oms = 2.25e-22
    S_sn = S_oms + S_acc
    return (20 / 3) * S_sn / (2 * math.pi * f) ** 2


def DPsi(f, z, A, n=N_GRU):
    pre = (1 + n) / (2 * H0)
    Eg = hP * f
    K = K_int(z, n)
    dt = pre * A * (Eg / EPl) ** n * K
    return -2 * math.pi * f * dt


def dPsidN(f, z, A, dn=1e-4):
    return (DPsi(f, z, A, N_GRU + dn) - DPsi(f, z, A, N_GRU - dn)) / (2 * dn)


def fisher_sigma_n(A):
    """Devuelve (I_total, sigma_n_1evento) para una amplitud A dada."""
    I_total = 0.0
    rows = []
    for fmhz in [0.3, 1.0, 3.0, 10.0, 30.0]:
        for z in [1.0, 2.0]:
            f = fmhz * 1e-3
            Sh = Sh_LISA(f)
            sp = math.sqrt(Sh / T_obs) / (1e-21)
            dpdn = dPsidN(f, z, A)
            Ic = dpdn ** 2 / (sp ** 2) if sp > 0 else 0
            I_total += Ic
            rows.append((fmhz, z, dpdn, Ic))
    sigma_n = 1 / math.sqrt(I_total) if I_total > 0 else float("inf")
    return I_total, sigma_n, rows


if __name__ == "__main__":
    print("=" * 70)
    print("GRU_P_FISHER_v3_OFICIAL — Recalculo con A_final=1.46e-16")
    print("=" * 70)

    I_total, sigma_n, rows = fisher_sigma_n(A_FINAL)
    sigma_n_safe = sigma_n * SAFETY_FACTOR
    N5 = max(1, (5 * sigma_n_safe / N_GRU) ** 2)

    print(f"\nA_final usado        = {A_FINAL:.4e}")
    print(f"I_total (Fisher)     = {I_total:.4e}")
    print(f"sigma_n (1 evento)   = {sigma_n:.6e}")
    print(f"sigma_n_safe (x{SAFETY_FACTOR:.0f})   = {sigma_n_safe:.6e}")
    print(f"N5 (eventos 5-sigma) = {N5:.4e}")

    # --- Separaciones para Sec. 7.4.8 (usa sigma_n SIN safety factor,
    #     igual que el texto original: "sigma_n=0.0027 ... 720 sigma") ---
    sep_n2 = abs(N_GRU - N_ALT_MASSIVE_GRAVITY_LQG) / sigma_n
    sep_n1 = abs(N_GRU - N_ALT_SME_D5) / sigma_n

    print("\n--- Separaciones de discriminabilidad (Sec. 7.4.8) ---")
    print(f"Separacion vs n=2 (masiva/LQG) = {sep_n2:.3e} sigma")
    print(f"Separacion vs n=1 (SME d=5)    = {sep_n1:.3e} sigma")
    print("\nNOTA: estos valores sustituyen los '720 sigma' y '349 sigma'")
    print("del HTML actual, que fueron calculados con sigma_n=0.0027")
    print("(amplitud pre-screening, incorrecta).")

    if sep_n2 < 5 and sep_n1 < 5:
        print("\n=> Con A_final, GRU NO logra separacion estadistica")
        print("   significativa (<5 sigma) de n=1 ni n=2 via LISA.")
        print("   La afirmacion de 'unique observable signature' con")
        print("   cifras de sigma debe eliminarse o reformularse en")
        print("   terminos puramente cualitativos (n_GRU<1, anisotropia).")

    result = {
        "script": "GRU_P_FISHER_v3_OFICIAL_A_FINAL.py",
        "fecha": "2026-07-05",
        "A_final_usado": A_FINAL,
        "n_GRU": N_GRU,
        "I_total": I_total,
        "sigma_n_1evento": sigma_n,
        "sigma_n_safe": sigma_n_safe,
        "N5_eventos_5sigma": N5,
        "separacion_sigma_vs_n2_masiva_LQG": sep_n2,
        "separacion_sigma_vs_n1_SME_d5": sep_n1,
        "nota": (
            "Sustituye sigma_n=0.0027 (con A pre-screening, INCORRECTO) "
            "usado en lineas 242, 1140, 1894 y en Sec. 7.4.8 del HTML v2.6."
        ),
    }
    with open("GRU_P_FISHER_v3_OFICIAL_A_FINAL_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nJSON guardado: GRU_P_FISHER_v3_OFICIAL_A_FINAL_result.json")
