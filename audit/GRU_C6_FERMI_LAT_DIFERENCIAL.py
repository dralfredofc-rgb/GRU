#!/usr/bin/env python3
"""
GRU_C6_FERMI_LAT_DIFERENCIAL.py
==================================================================
Recomputo de la cota Fermi-LAT (GRB 090510) sobre A_final usando la
forma DIFERENCIAL correcta del retardo temporal MDR:

    Delta_t = (1/2)(1-n) * A * [ (E_high/E_Pl)^n - (E_low/E_Pl)^n ] * D(z,n) / H0

donde D(z,n) = H0 * Integral_0^z (1+z')^n / H(z') dz'  es ADIMENSIONAL.

Corrige dos errores previos identificados en sesión:
  1) Usar (E_high/E_Pl)^n de forma ABSOLUTA en vez de la diferencia
     (E_high^n - E_low^n) — en n=1 es irrelevante, en n=0.0564 no lo es
     (el término de baja energía es ~70% del alto).
  2) Confundir D(z,n) adimensional (=H0*Integral) con Integral en
     segundos en el denominador (error de factor ~1e18 encontrado y
     corregido en sesión del 8 jul 2026).

Verificado independientemente: coincide con calculo hecho por Claude
en sesión (M1+100MeV: A_max=2.0015e-16, ratio=0.729) y por Hermes/Kimi
(mismos números, ver GRU_C6_FERMI_LAT_DIFERENCIAL_v1_3.json).

Uso: python3 GRU_C6_FERMI_LAT_DIFERENCIAL.py
==================================================================
"""
import numpy as np
from scipy.integrate import quad
import json

# ------------------------------------------------------------------
# CONSTANTES
# ------------------------------------------------------------------
E_Pl_GeV = 1.2209e19
H0_km_s_Mpc = 72.0
Mpc_to_km = 3.08567758e19
H0_s = H0_km_s_Mpc / Mpc_to_km          # s^-1

Omega_m, Omega_L = 0.3, 0.7
z_GRB090510 = 0.903
n_GRU = 0.0564
A_final_GRU = 1.46e-16
E_high_GeV = 31.0

# Bandas de referencia de baja energía candidatas
E_LOW_BANDS = {
    "100_MeV_LAT": 0.1,
    "1_MeV_GBM": 0.001,
    "100_keV": 0.0001,
}

# Métodos de timing (Delta t), de Vasileiou 2013 / Granot 2015
METHODS = {
    "M1_conservador": 0.829,   # t_start conservador, todo el burst
    "M2_spike": 0.010,         # asociación al spike principal
    "M3_DisCan": 0.93,         # método DisCan, |dt/dE|<30 ms/GeV, ~99% CL
}


def D_z_n(z, n, Om=Omega_m, OL=Omega_L):
    """D(z,n) = H0 * Integral_0^z (1+z')^n / H(z') dz'  (adimensional)."""
    integrand = lambda zp: (1 + zp) ** n / np.sqrt(Om * (1 + zp) ** 3 + OL)
    I_seconds = quad(integrand, 0, z)[0] / H0_s
    return H0_s * I_seconds, I_seconds  # (adimensional, en segundos)


def A_max_differential(delta_t_s, E_low_GeV, n=n_GRU, z=z_GRB090510):
    D, I_sec = D_z_n(z, n)
    term_high = (E_high_GeV / E_Pl_GeV) ** n
    term_low = (E_low_GeV / E_Pl_GeV) ** n
    delta_term = term_high - term_low
    numerator = 2 * H0_s * delta_t_s
    denominator = (1 - n) * delta_term * D
    A_max = numerator / denominator
    return A_max, {
        "D_z_n": D, "term_high": term_high, "term_low": term_low,
        "delta_term": delta_term, "numerator": numerator,
        "denominator": denominator,
    }


def A_max_absolute(delta_t_s, n=n_GRU, z=z_GRB090510):
    """Forma absoluta (referencia histórica / error de convención)."""
    D, _ = D_z_n(z, n)
    term = (E_high_GeV / E_Pl_GeV) ** n
    numerator = 2 * H0_s * delta_t_s
    denominator = (1 - n) * term * D
    return numerator / denominator


if __name__ == "__main__":
    print("=" * 70)
    print("GRU C6 — Recomputo diferencial Fermi-LAT GRB 090510")
    print("=" * 70)
    print(f"z={z_GRB090510}, E_high={E_high_GeV} GeV, n_GRU={n_GRU}, A_final={A_final_GRU:.3e}")
    print(f"H0 = {H0_s:.6e} s^-1")

    results = {"differential_form": {}, "absolute_form": {}}

    for band_name, E_low in E_LOW_BANDS.items():
        for method_name, dt in METHODS.items():
            A_max, details = A_max_differential(dt, E_low)
            ratio = A_final_GRU / A_max
            verdict = "VIOLA" if ratio > 1.0 else ("TENSION" if ratio > 0.9 else "PASA")
            key = f"{band_name}__{method_name}"
            results["differential_form"][key] = {
                "band_name": band_name, "E_low_GeV": E_low,
                "method": method_name, "delta_t_s": dt,
                "A_max": A_max, "A_final_over_A_max": ratio,
                "verdict": verdict, "details": details,
            }
            print(f"  {key:35s} A_max={A_max:.4e}  ratio={ratio:6.3f}  {verdict}")

    print("\n--- Forma absoluta (referencia, subestima la cota) ---")
    for method_name, dt in METHODS.items():
        A_max = A_max_absolute(dt)
        ratio = A_final_GRU / A_max
        verdict = "VIOLA" if ratio > 1.0 else "PASA"
        results["absolute_form"][method_name] = {
            "method": method_name, "delta_t_s": dt,
            "A_max": A_max, "A_final_over_A_max": ratio, "verdict": verdict,
        }
        print(f"  {method_name:20s} A_max={A_max:.4e}  ratio={ratio:6.3f}  {verdict}")

    results["metadata"] = {
        "script": "GRU_C6_FERMI_LAT_DIFERENCIAL.py",
        "date": "2026-07-08",
        "n_GRU": n_GRU, "A_final_GRU": A_final_GRU, "z": z_GRB090510,
        "note": (
            "Forma diferencial (E_high^n - E_low^n) es la fisicamente correcta "
            "para n fraccionario; la forma absoluta sobreestima la cota (subestima "
            "A_max) porque ignora que a n=0.0564 el termino de baja energia es "
            "~50-72% del termino de alta energia, cosa irrelevante en n=1 pero "
            "decisiva aqui."
        ),
    }

    with open("GRU_C6_FERMI_LAT_DIFERENCIAL_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nJSON guardado: GRU_C6_FERMI_LAT_DIFERENCIAL_result.json")
    print("\nCONCLUSION: bajo M1 (conservador, banda 100 MeV), A_final PASA")
    print("la cota (ratio<1). Bajo M2 (asociacion a spike, menos conservador),")
    print("VIOLA con margen. La banda de referencia y el metodo de timing")
    print("cambian el veredicto -> reportar como TENSION MARGINAL, no como")
    print("'violacion robusta' ni como 'compatible sin mas'.")
