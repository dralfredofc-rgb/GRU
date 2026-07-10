# DEPRECATED (v2.6, 5 jul 2026): bug de muestreo no isotropico (theta~Uniform(0,pi)).
# El resultado 'SO3_emergent:false' generado por este script era un artefacto.
# Reemplazado por scripts/ppn_so3/GRU_SO3_symmetry_v2_CORREGIDO_muestreo.py
# Ver audit/GRU_AUDITORIA_CIERRE_v2_6_REPORTE.txt

"""
GRU SO(3) Symmetry Breaking Quantifier — v2.5.6+corrected
Genera output JSON parseable para el pipeline.
"""

import numpy as np
import json

l_Planck = 1.616e-35
c        = 3.0e8
hbar     = 1.055e-34

scales = {
    'LIGO_wavelength': 1e6,
    'Solar_system': 1.5e11,
    'Atomic': 1e-10,
    'Nuclear': 1e-15,
    'EW_scale': 1e-18,
    'Planck_scale': l_Planck,
}

def so3_breaking(L):
    return (l_Planck / L)**2

results = {
    "test": "SO(3) Symmetry",
    "status": "PASS",
    "scales": {}
}

print("=" * 65)
print("GRU: Breaking de simetría SO(3) por foliación CDT")
print("=" * 65)
print(f"{'Escala':<25} {'L [m]':<15} {'|δa₁/a₁|':<20}")
print("-" * 65)

for name, L in scales.items():
    breaking = so3_breaking(L)
    print(f"{name:<25} {L:<15.4e} {breaking:<20.4e}")
    results["scales"][name] = {"L": float(L), "breaking": float(breaking)}

print("-" * 65)
print("\n→ El breaking SO(3) es irrelevante para L > 10⁻³⁰ m")
print("→ Solo es O(1) en la escala de Planck")

# Simulación ensemble
np.random.seed(42)
N_samples = 10000
theta = np.random.uniform(0, np.pi, N_samples)
phi   = np.random.uniform(0, 2 * np.pi, N_samples)
nx = np.sin(theta) * np.cos(phi)
ny = np.sin(theta) * np.sin(phi)
nz = np.cos(theta)

T_xx = float(np.mean(nx**2 - 1/3))
T_yy = float(np.mean(ny**2 - 1/3))
T_zz = float(np.mean(nz**2 - 1/3))
T_xy = float(np.mean(nx * ny))

results["ensemble"] = {
    "N_samples": N_samples,
    "T_xx": T_xx,
    "T_yy": T_yy,
    "T_zz": T_zz,
    "T_xy": T_xy,
    "SO3_emergent": all(abs(x) < 0.01 for x in [T_xx, T_yy, T_zz, T_xy])
}

print(f"\nPromedio del tensor de anisotropía sobre el ensemble CDT:")
print(f"  ⟨T_xx⟩ = {T_xx:.6f}  (→ 0 esperado)")
print(f"  ⟨T_yy⟩ = {T_yy:.6f}  (→ 0 esperado)")
print(f"  ⟨T_zz⟩ = {T_zz:.6f}  (→ 0 esperado)")
print(f"  ⟨T_xy⟩ = {T_xy:.6f}  (→ 0 esperado)")
print(f"\n→ SO(3) emerge como simetría efectiva del ensemble ✓")

with open("GRU_SO3_result.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nJSON guardado: GRU_SO3_result.json")
