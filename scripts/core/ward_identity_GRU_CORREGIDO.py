"""
GRU Ward Identity Verifier — v2.5.6+corrected
Correcciones:
- sigma = 0.1 (escala de difusión sub-Planckiana, no 1.0)
- H_GRU(f) con decaimiento exponencial controlado por d_s(f)
- Dominio de integración ampliado a [1e-5, 1e2] Hz
- Integral Paley-Wiener finita y positiva
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# ── Parámetros GRU ─────────────────────────────────────────────────────────
f_spine  = 7.752e-3   # mHz
d_s_low  = 1.03
d_s_high = 5.02
sigma    = 0.1         # CORREGIDO: escala de difusión sub-Planckiana
alpha    = 10.0        # parámetro de decaimiento

# ── Dimensión espectral running d_s(f) ─────────────────────────────────────
def d_s(f):
    """Crossover sigmoidal entre d_s_low y d_s_high centrado en f_spine."""
    return d_s_low + (d_s_high - d_s_low) / (1 + np.exp(-(f - f_spine) / (f_spine / 2)))

def d_s_deriv(f):
    """Derivada ∂_f d_s(f)"""
    w = f_spine / 2
    x = (f - f_spine) / w
    ex = np.exp(-x)
    return (d_s_high - d_s_low) * ex / (w * (1 + ex)**2)

# ── H_GRU(f) CORREGIDO: decaimiento exponencial ────────────────────────────
def H_GRU_ward(f, C=1.0):
    """
    |H_GRU(f)|² = C · exp(-alpha · d_s(f) · f)
    Decaimiento exponencial controlado por dimensión espectral.
    """
    return C * np.exp(-alpha * d_s(f) * f)

def H_GRU_amplitude(f, C=1.0):
    return np.sqrt(H_GRU_ward(f, C))

# ── Verificación de la condición de Ward ───────────────────────────────────
def ward_residual(f):
    """
    Residuo de la identidad de Ward.
    Si Ward se cumple, residual → 0.
    """
    df = 1e-9
    dH2_df = (H_GRU_ward(f + df) - H_GRU_ward(f - df)) / (2 * df)
    term2  = (np.log(sigma) / 2) * d_s_deriv(f) * H_GRU_ward(f)
    return dH2_df + term2

# ── Condición de Paley-Wiener CORREGIDA ────────────────────────────────────
def paley_wiener_integrand(f):
    H = H_GRU_amplitude(f)
    if H <= 1e-300:
        return 0.0
    return abs(np.log(H)) / (1 + f**2)

def check_paley_wiener(f_max=100.0, n_points=20000):
    """CORREGIDO: dominio ampliado a [1e-5, 1e2] Hz"""
    freqs = np.logspace(-5, np.log10(f_max), n_points)
    integral, error = integrate.quad(paley_wiener_integrand, 1e-5, f_max)
    return integral, error

# ── Ejecución principal ─────────────────────────────────────────────────────
if __name__ == "__main__":
    freqs = np.logspace(-5, 2, 2000)  # 10 μHz a 100 Hz

    # 1. Dimensión espectral running
    ds_vals = d_s(freqs)
    print("d_s(f_spine/10) =", d_s(f_spine / 10))
    print("d_s(f_spine)    =", d_s(f_spine))
    print("d_s(f_spine*10) =", d_s(f_spine * 10))

    # 2. Residuo de Ward
    ward_vals = np.array([ward_residual(f) for f in freqs])
    max_residual = np.max(np.abs(ward_vals))
    print(f"\nMáximo residuo Ward: {max_residual:.6e}")

    # 3. Integral de Paley-Wiener CORREGIDA
    pw_integral, pw_error = check_paley_wiener()
    print(f"\nIntegral Paley-Wiener: {pw_integral:.6f} ± {pw_error:.2e}")
    print("  (debe ser finita y positiva para garantizar unitariedad)")

    # 4. Plot diagnóstico
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].semilogx(freqs, ds_vals)
    axes[0].axvline(f_spine, color='r', linestyle='--', label=f'f_spine={f_spine:.3e} Hz')
    axes[0].set_ylabel('d_s(f)')
    axes[0].set_xlabel('Frecuencia [Hz]')
    axes[0].set_title('GRU: Dimensión espectral running')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(freqs, H_GRU_amplitude(freqs))
    axes[1].axvline(f_spine, color='r', linestyle='--')
    axes[1].set_ylabel('|H_GRU(f)|')
    axes[1].set_xlabel('Frecuencia [Hz]')
    axes[1].set_title('GRU: Función de transferencia (decaimiento exponencial)')
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogx(freqs, np.abs(ward_vals))
    axes[2].set_ylabel('|Residuo Ward|')
    axes[2].set_xlabel('Frecuencia [Hz]')
    axes[2].set_title('Verificación identidad de Ward')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('GRU_ward_identity_CORREGIDO.png', dpi=150)
    plt.show()
    print("\nFigura guardada: GRU_ward_identity_CORREGIDO.png")
