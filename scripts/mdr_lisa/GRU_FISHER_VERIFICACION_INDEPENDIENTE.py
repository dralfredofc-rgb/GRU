#!/usr/bin/env python3
"""
GRU Fisher information — VERIFICACIÓN INDEPENDIENTE
==================================================
Script standalone de auditoría. No modifica archivos existentes.

Mantiene la matemática original de GRU_P_FISHER_v2_FIXED.py:
- K_int(), Sh_LISA(), DPsi(), dPsidN() copiadas tal cual
- grid de frecuencias: [0.3, 1.0, 3.0, 10.0, 30.0] mHz
- redshifts: [1.0, 2.0]
- safety factor: 10x
"""
import math
import os
import json

try:
    import numpy as np
except ImportError:
    raise SystemExit("numpy requerido para este script")

try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz

H0 = 70e3 / 3.086e22
EPl = 1.956e9 * 1.602e-10
hP = 6.626e-34
Om, OL = 0.30, 0.70
N_GRU = 0.0564

T_obs = 4 * 365.25 * 24 * 3600
SAFETY_FACTOR = 10.0


def K_int(z, n=N_GRU, N=800):
    zz = np.linspace(0, z, N)
    return trapz((1 + zz) ** n / np.sqrt(Om * (1 + zz) ** 3 + OL), zz)


def Sh_LISA(f):
    """Curva de ruido LISA aproximada (L3S 2024)."""
    S_acc = 9e-30 / (2 * math.pi * f) ** 4
    S_oms = 2.25e-22
    S_sn = S_oms + S_acc
    return (20 / 3) * S_sn / (2 * math.pi * f) ** 2


def DPsi(f, z, n=N_GRU, A=1.0e-17):
    pre = (1 + n) / (2 * H0)
    Eg = hP * f
    K = K_int(z, n)
    dt = pre * A * (Eg / EPl) ** n * K
    return -2 * math.pi * f * dt


def dPsidN(f, z, dn=1e-4, A=1.0e-17):
    return (DPsi(f, z, N_GRU + dn, A) - DPsi(f, z, N_GRU - dn, A)) / (2 * dn)


def compute_case(A):
    I_total = 0.0
    rows = []
    for fmhz in [0.3, 1.0, 3.0, 10.0, 30.0]:
        for z in [1.0, 2.0]:
            f = fmhz * 1e-3
            Sh = Sh_LISA(f)
            sp = math.sqrt(Sh / T_obs) / (1e-21)
            dpdn = dPsidN(f, z, A=A)
            Ic = dpdn ** 2 / (sp ** 2) if sp > 0 else 0.0
            I_total += Ic
            sn = 1.0 / math.sqrt(abs(Ic)) if Ic > 0 else float("inf")
            sn_safe = sn * SAFETY_FACTOR
            rows.append({
                "f_mHz": fmhz,
                "z": z,
                "dPsi_dN": dpdn,
                "sigma_n": sn,
                "sigma_n_safe": sn_safe,
                "I_cell": Ic,
            })
    sn_1 = 1.0 / math.sqrt(abs(I_total)) if I_total > 0 else float("inf")
    sn_1_safe = sn_1 * SAFETY_FACTOR
    N5 = max(1, (5 * sn_1_safe / N_GRU) ** 2) if N_GRU > 0 else float("inf")
    return {
        "A": A,
        "I_total": I_total,
        "sigma_n": sn_1,
        "sigma_n_safe": sn_1_safe,
        "N5": N5,
        "rows": rows,
    }


def main():
    valores_A = [
        1.9148e-08,
        1.46e-16,
        4.235e-08,
    ]

    resultados = []
    for A in valores_A:
        resultados.append(compute_case(A))

    salida = {
        "N_GRU": N_GRU,
        "T_obs_years": T_obs / (365.25 * 24 * 3600),
        "safety_factor": SAFETY_FACTOR,
        "resultados": resultados,
    }

    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GRU_FISHER_VERIFICACION_INDEPENDIENTE_result.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    # Tabla comparativa
    print("=" * 90)
    print("GRU Fisher verification — resultados comparativos")
    print("=" * 90)
    print(f"{'A':<15} {'I_total':<15} {'sigma_n':<15} {'sigma_n_safe':<15} {'N5':<15}")
    print("-" * 90)
    for r in resultados:
        print(
            f"{r['A']:<15.4e} {r['I_total']:<15.5e} {r['sigma_n']:<15.6f} {r['sigma_n_safe']:<15.6f} {r['N5']:<15.1f}"
        )
    print("-" * 90)

    # Escalamiento 1/A
    base = resultados[0]
    print("\nVerificación explícita: ratio sigma_n ~ 1/A ?")
    print(f"{'Par':<40} {'Esperado':<15} {'Medido':<15}")
    print("-" * 90)
    for i in range(1, len(resultados)):
        actual = resultados[i]
        ratio_A = base["A"] / actual["A"]
        ratio_sigma = base["sigma_n"] / actual["sigma_n"]
        ratio_sigma_safe = base["sigma_n_safe"] / actual["sigma_n_safe"]
        ratio_I = actual["I_total"] / base["I_total"]
        print(
            f"{'sigma_n ' + str(i):<40} {ratio_A:<15.6f} {ratio_sigma:<15.6f} | "
            f"sigma_n_safe ratio={ratio_sigma_safe:.6f}, I ratio={ratio_I:.6f}"
        )
    print("=" * 90)
    print("Resultados guardados en:", ruta)


if __name__ == "__main__":
    main()
