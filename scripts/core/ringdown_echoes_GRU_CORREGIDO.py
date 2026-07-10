"""
GRU Ringdown Echo Calculator — v2.5.6+corrected
Correcciones:
- Unidades: Delta_t en segundos (no ms)
- Etiquetas de salida corregidas
- NOTA explícita sobre escala temporal de ecos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, integrate

# ── Parámetros GRU ─────────────────────────────────────────────────────────
f_spine  = 7.752e-3   # mHz
Delta_t  = 1 / f_spine  # 128.999 SEGUNDOS (CORREGIDO: no ms)
A_GRU    = 4.235e-8
n_GRU    = 0.0564
d_s_low  = 1.03
d_s_high = 5.02

# ── Constantes físicas ────────────────────────────────────────────────────
G      = 6.674e-11
c      = 3.0e8
M_sun  = 1.989e30
l_P    = 1.616e-35

# ── Dimensión espectral y H_GRU ────────────────────────────────────────────
def d_s(omega):
    f = omega / (2 * np.pi)
    return d_s_low + (d_s_high - d_s_low) / (1 + np.exp(-(f - f_spine) / (f_spine / 2)))

def H_GRU_modulus(omega):
    ds_val = d_s(omega)
    ds_ref = d_s_high
    return np.exp(-(ds_val - ds_ref) / 2 * np.log(2))

# ── Potencial de Regge-Wheeler ──────────────────────────────────────────────
def V_RW(r, M, l=2):
    rs = 2 * G * M / c**2
    f  = 1 - rs / r
    return f * (l * (l + 1) / r**2 - 3 * rs / (2 * r**3))

# ── Potencial GRU modificado ─────────────────────────────────────────────────
def V_GRU_barrier(r, M, amplitude=0.1):
    rs = 2 * G * M / c**2
    epsilon_QG = (l_P / rs)**0.5
    r_barrier  = rs * (1 + 10 * epsilon_QG)
    w_barrier  = Delta_t * c / (4 * np.pi)
    return amplitude * np.exp(-0.5 * ((r - r_barrier) / w_barrier)**2) / r**2

def V_eff_GRU(r, M, omega, barrier_amplitude=0.1):
    H2   = H_GRU_modulus(omega)
    V_rw = V_RW(r, M)
    V_b  = V_GRU_barrier(r, M, barrier_amplitude)
    return V_rw * H2 + V_b * (1 - H2)

# ── Retardo de eco GRU ─────────────────────────────────────────────────────
def echo_delay_GRU(M_solar):
    M  = M_solar * M_sun
    rs = 2 * G * M / c**2
    Delta_t_geom = 4 * rs / c
    return Delta_t_geom + Delta_t, Delta_t_geom, Delta_t

# ── Cálculo para masas de 5-100 M_sun ─────────────────────────────────────
masses = np.array([5, 10, 20, 30, 50, 70, 100])

print("=" * 70)
print("GRU: Predicción de retardo de ecos en ringdown de agujeros negros")
print("=" * 70)
print(f"Δt_GRU universal = {Delta_t:.3f} s  (= 1/f_spine)")
print("NOTA: Este retardo es la escala característica del SGWB en LISA.")
print("      Para ecos en ringdown de LIGO, ver sección 'Escala temporal'.")
print()
print(f"{'M [M☉]':<10} {'Δt_geom [ms]':<18} {'Δt_GRU [s]':<18} {'Δt_total [s]':<18} {'GRU domina?'}")
print("-" * 70)

for M_s in masses:
    dt_total, dt_geom, dt_gru = echo_delay_GRU(M_s)
    gru_dominant = "SÍ ✓" if dt_gru > dt_geom else "NO ✗"
    print(f"{M_s:<10} {dt_geom*1e3:<18.2f} {dt_gru:<18.3f} {dt_total:<18.3f} {gru_dominant}")

print("-" * 70)
print("\n→ Para M < 1000 M☉, el retardo GRU DOMINA sobre el geométrico")
print(f"→ Señal predicha: tren de ecos con separación fija ~{Delta_t:.0f} s")

# ── SECCIÓN: Escala temporal de ecos ───────────────────────────────────────
print("\n" + "=" * 70)
print("NOTA IMPORTANTE SOBRE ESCALA TEMPORAL DE ECOS")
print("=" * 70)
print(f"""
El retardo GRU Δt = {Delta_t:.3f} s es la escala característica de la
espina causal CDT. Para el SGWB en LISA (frecuencias ~mHz), esta escala
es la correcta.

Para ecos en ringdown de fusiones de agujeros negros (LIGO):
- Ringdown duration τ ~ 0.5 s (M = 30 M⊙)
- Δt_GRU = {Delta_t:.0f} s >> τ_ringdown

Esto significa que un eco a {Delta_t:.0f} s del merger estaría en el
ruido de LIGO. La predicción de ecos con esta escala es, por tanto,
una hipótesis especulativa pendiente de refinamiento matemático.

POSIBLES REFINAMIENTOS (v2.7+):
a) Mecanismo de resonancia que comprima la escala a ~0.1-1 s
b) Múltiples reflexiones con retardo acumulado N × Δt_corto
c) Escala efectiva de eco independiente de 1/f_spine
""")

# ── Diagnóstico del potencial doble pozo ──────────────────────────────────
M_test  = 30 * M_sun
rs_test = 2 * G * M_test / c**2

r_vals  = np.linspace(1.01 * rs_test, 15 * rs_test, 10000)
omega_RW   = 2 * np.pi * 200
omega_GRU  = 2 * np.pi * f_spine

V_classical = np.array([V_RW(r, M_test) for r in r_vals])
V_at_fspine = np.array([V_eff_GRU(r, M_test, omega_GRU) for r in r_vals])

# Buscar doble pozo
dV = np.gradient(V_at_fspine, r_vals)
sign_changes = np.where(np.diff(np.sign(dV)))[0]

print(f"\nAnálisis de doble pozo para M = 30 M☉:")
print(f"  r_s = {rs_test/1e3:.1f} km")
print(f"  Número de puntos críticos en V_eff_GRU: {len(sign_changes)}")
for i, idx in enumerate(sign_changes[:5]):
    r_crit = r_vals[idx]
    V_crit = V_at_fspine[idx]
    tipo   = "máximo" if dV[idx] > 0 else "mínimo"
    print(f"  Punto crítico {i+1}: r = {r_crit/rs_test:.2f} r_s, V = {V_crit:.4e}, tipo: {tipo}")

# ── Plot del potencial ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
r_norm = r_vals / rs_test

ax.plot(r_norm, V_classical * rs_test**2, 'b-', lw=2, label='V_RW clásico')
ax.plot(r_norm, V_at_fspine * rs_test**2, 'r-', lw=2, label=f'V_GRU (ω = ω_spine)')
ax.axvline(1.0, color='k', linestyle=':', alpha=0.5, label='Horizonte r_s')
ax.axvline(3.0, color='g', linestyle='--', alpha=0.5, label='Fotosfera 3r_s')
ax.set_xlim(1, 15)
ax.set_ylim(-0.01, 0.05)
ax.set_xlabel('r / r_s', fontsize=12)
ax.set_ylabel('V · r_s²', fontsize=12)
ax.set_title(f'Potencial efectivo GRU vs Regge-Wheeler\nM = 30 M☉, f_spine = {f_spine:.3f} Hz', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('GRU_ringdown_potential_CORREGIDO.png', dpi=150)
plt.show()
print("\nFigura guardada: GRU_ringdown_potential_CORREGIDO.png")

# ── Template de búsqueda de ecos ───────────────────────────────────────────
def echo_template_GRU(t, t_merger, Delta_t_echo, n_echoes=5,
                       tau_decay=0.5, omega_qnm=2*np.pi*100):
    h = np.zeros_like(t)
    for n in range(n_echoes):
        t_echo = t_merger + (n + 1) * Delta_t_echo
        mask   = t > t_echo
        amplitude_n = np.exp(-n * Delta_t_echo / tau_decay)
        h[mask] += amplitude_n * np.exp(-(t[mask] - t_echo) / tau_decay)                    * np.cos(omega_qnm * (t[mask] - t_echo))
    return h

# Ejemplo con Delta_t = 0.129 s (escala hipotética para LIGO)
t = np.linspace(0, 2.0, 200000)
Delta_t_LIGO = 0.129  # s, escala hipotética para ecos en LIGO
h_echo = echo_template_GRU(t, t_merger=0.1, Delta_t_echo=Delta_t_LIGO)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t, h_echo, 'b-', lw=0.5)
ax.axvline(0.1, color='r', linestyle='--', label='Merger')
for n in range(1, 6):
    ax.axvline(0.1 + n * Delta_t_LIGO, color='g', linestyle=':', alpha=0.7,
               label=f'Eco {n}' if n == 1 else None)
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('h(t) [normalizado]')
ax.set_title(f'Template GRU: ecos con Δt = {Delta_t_LIGO*1e3:.1f} ms (escala hipotética LIGO)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('GRU_echo_template_CORREGIDO.png', dpi=150)
plt.show()
print("Figura guardada: GRU_echo_template_CORREGIDO.png")
