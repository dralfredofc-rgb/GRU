#!/usr/bin/env python3
"""
GRU_SO3_symmetry_v2_CORREGIDO_muestreo.py
==================================================================
Corrige so3_symmetry_GRU_CORREGIDO.py (parte del ZIP v2.5 publicado).

BUG ORIGINAL:
  theta = np.random.uniform(0, np.pi, N_samples)
Esto NO produce puntos isotropicamente distribuidos en la esfera: la
medida correcta es dOmega = sin(theta) dtheta dphi, y muestrear theta
uniforme sobresample los polos. El resultado "SO3_emergent: false" del
JSON publicado es consecuencia analitica de este bug (verificado:
E[T_zz]=1/6, E[T_xx]=E[T_yy]=-1/12 bajo la distribucion incorrecta),
no una medicion fisica real de datos CDT/GRU.

CORRECCION:
  theta = np.arccos(np.random.uniform(-1, 1, N_samples))
==================================================================
"""
import numpy as np
import json

l_Planck = 1.616e-35
c = 3.0e8
hbar = 1.055e-34

scales = {
    "LIGO_wavelength": 1e6,
    "Solar_system": 1.5e11,
    "Atomic": 1e-10,
    "Nuclear": 1e-15,
    "EW_scale": 1e-18,
    "Planck_scale": l_Planck,
}


def so3_breaking(L):
    return (l_Planck / L) ** 2


results = {
    "test": "SO(3) Symmetry (muestreo corregido)",
    "status": "PASS",
    "scales": {},
    "bug_previo": (
        "so3_symmetry_GRU_CORREGIDO.py usaba theta~Uniform(0,pi); "
        "corregido a cos(theta)~Uniform(-1,1) para muestreo isotropico real."
    ),
}

print("=" * 65)
print("GRU: Breaking de simetria SO(3) por foliacion CDT (muestreo FIX)")
print("=" * 65)
print(f"{'Escala':<25} {'L [m]':<15} {'|delta a1/a1|':<20}")
print("-" * 65)

for name, L in scales.items():
    breaking = so3_breaking(L)
    print(f"{name:<25} {L:<15.4e} {breaking:<20.4e}")
    results["scales"][name] = {"L": float(L), "breaking": float(breaking)}

print("-" * 65)
print("-> El breaking SO(3) es irrelevante para L > 1e-30 m")
print("-> Solo es O(1) en la escala de Planck")
print("(Esta parte NO cambia respecto al script original: es analitica,")
print(" no depende del muestreo del ensemble.)")

# --- Ensemble CORREGIDO: muestreo isotropico real sobre la esfera ---
np.random.seed(42)
N_samples = 10000
theta = np.arccos(np.random.uniform(-1, 1, N_samples))   # <-- CORREGIDO
phi = np.random.uniform(0, 2 * np.pi, N_samples)
nx = np.sin(theta) * np.cos(phi)
ny = np.sin(theta) * np.sin(phi)
nz = np.cos(theta)

T_xx = float(np.mean(nx ** 2 - 1 / 3))
T_yy = float(np.mean(ny ** 2 - 1 / 3))
T_zz = float(np.mean(nz ** 2 - 1 / 3))
T_xy = float(np.mean(nx * ny))

results["ensemble"] = {
    "N_samples": N_samples,
    "T_xx": T_xx,
    "T_yy": T_yy,
    "T_zz": T_zz,
    "T_xy": T_xy,
    "SO3_emergent": all(abs(x) < 0.01 for x in [T_xx, T_yy, T_zz, T_xy]),
}

print(f"\nPromedio del tensor de anisotropia sobre el ensemble (muestreo FIX):")
print(f"  <T_xx> = {T_xx:.6f}  (-> 0 esperado)")
print(f"  <T_yy> = {T_yy:.6f}  (-> 0 esperado)")
print(f"  <T_zz> = {T_zz:.6f}  (-> 0 esperado)")
print(f"  <T_xy> = {T_xy:.6f}  (-> 0 esperado)")
print(f"\nSO3_emergent = {results['ensemble']['SO3_emergent']}")

with open("GRU_SO3_result_CORREGIDO.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nJSON guardado: GRU_SO3_result_CORREGIDO.json")
print("\nIMPORTANTE: este ensemble sigue sin usar datos CDT/GRU reales,")
print("es una prueba sintetica de muestreo isotropico. Si se quiere una")
print("medicion fisica real de isotropia del spine, hay que construirla")
print("a partir de un ensemble de geometrias CDT reales, no de puntos")
print("aleatorios sinteticos.")
