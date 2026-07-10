#!/usr/bin/env python3
"""
GRU-LISA: Stochastic Gravitational Wave Background Spectrum
============================================================

Generates the predicted S_h(f) curve with GRU spectral break and compares
against standard SGWB models (inflation, cosmic strings, etc.).

GRU prediction:
- Below f_c (spine-dominated): S_h ~ f^0 (flat, d_s=1)
- Above f_c (bulk-suppressed): S_h ~ f^{-2} (d_s=1.67 mapped to power-law)
- Break at f_c determined by spine correlation length

Reference: GRU v2.4 LISA phenomenology (memorandum §A.41-A.43)
            3B1B Fourier Transform (spUNpyF58BY) — spectral resolution

Usage:
    python3 GRU_LISA_SGWB_SPECTRUM.py --f_screen 0.50 --mode plot
    python3 GRU_LISA_SGWB_SPECTRUM.py --f_screen 0.50 --mode sensitivity
    python3 GRU_LISA_SGWB_SPECTRUM.py --f_screen 0.50 --mode compare
"""

import argparse
import numpy as np
import json
from datetime import datetime

# ── PHYSICAL CONSTANTS ──────────────────────────────────────────────────────

# LISA band (approximate)
F_LISA_MIN = 1e-4   # Hz
F_LISA_MAX = 1.0    # Hz
F_LISA_PEAK = 3e-3  # Hz (peak sensitivity)

# GRU parameters (from v2.4)
D_SPINE = 1.03      # d_s(spine) ~ 1.0
D_FULL = 1.67       # d_s(full) ~ 1.67 (from T1.3 decomposition)
A_GRU_DERIVED = 4.235e-8  # Derived amplitude (pre-screening)
LAMBDA_BARE = 0.50  # Cosmological constant
KAPPA_4D = 1.0 / (8.0 * np.pi)  # Einstein coupling D=4

# Standard SGWB models (power-law indices)
MODELS = {
    'inflation': {'alpha': -2.0, 'label': 'Inflation (slow-roll)'},
    'cosmic_strings': {'alpha': -3.0, 'label': 'Cosmic Strings'},
    'domain_walls': {'alpha': 1.0, 'label': 'Domain Walls'},
    'phase_transition': {'alpha': 3.0, 'label': '1st Order Phase Transition'},
    'gru_spine': {'alpha': 0.0, 'label': 'GRU Spine (d_s=1)'},
    'gru_bulk': {'alpha': -2.0, 'label': 'GRU Bulk (d_s=1.67)'},
}

# ── GRU SPECTRAL MODEL ──────────────────────────────────────────────────────

def gru_transfer_function(f, f_c, d_s_spine=1.03, d_s_bulk=1.67):
    """
    GRU spectral transfer function: low-pass filter from spine to bulk.

    Below f_c: spine dominates (flat spectrum, d_s ~ 1)
    Above f_c: bulk suppressed (power-law fall, d_s ~ 1.67)

    The transition is smooth (error-function) to avoid unphysical artifacts.
    """
    # Spine contribution: flat in f (alpha=0)
    spine_amp = 1.0

    # Bulk contribution: suppressed above f_c
    # Map d_s to power-law: P ~ f^{-(d_s - 1)} for d_s > 1
    bulk_exponent = -(d_s_bulk - d_s_spine)  # -(1.67 - 1.03) = -0.64
    bulk_amp = (f / f_c) ** bulk_exponent

    # Smooth transition
    x = np.log10(f / f_c)
    transition = 0.5 * (1.0 - np.tanh(2.0 * x))  # 1 below f_c, 0 above

    # Combined: spine below, bulk above with suppression
    H = transition * spine_amp + (1.0 - transition) * bulk_amp * 0.1

    return H

def gru_spectrum(f, A_gru, f_c, f_screen=1.0):
    """
    GRU SGWB spectrum: Omega_GW(f) = A_gru * H_GRU(f) * f_screen

    f_screen: screening factor (placeholder, to be calibrated with T-scan 3D)
    """
    H = gru_transfer_function(f, f_c)
    Omega = A_gru * H * f_screen
    return Omega

def standard_model_spectrum(f, A, alpha, f_ref=1e-3):
    """Standard power-law SGWB: Omega ~ A * (f/f_ref)^alpha"""
    return A * (f / f_ref) ** alpha

# ── LISA SENSITIVITY ────────────────────────────────────────────────────────

def lisa_sensitivity_curve(f):
    """
    Approximate LISA sensitivity curve (simplified analytic model).
    Based on L3ST sensitivity: https://lisa.nasa.gov
    """
    # Simplified model: minimum at ~3 mHz, rising at low and high f
    f_peak = 3e-3
    # Low-f rise: confusion noise from galactic binaries
    S_low = 1e-44 * (f / f_peak) ** (-2.0)
    # High-f rise: instrumental noise
    S_high = 1e-44 * (f / f_peak) ** 2.0
    # Flat minimum
    S_min = 1e-44

    S = np.maximum(S_low, np.maximum(S_high, S_min))
    return S

def omega_to_strain(Omega, f):
    """Convert Omega_GW to characteristic strain h_c."""
    # h_c^2 = (3 * H_0^2) / (2 * pi^2) * Omega_GW / f^2
    # Simplified: h_c ~ sqrt(Omega) / f
    H0 = 70.0  # km/s/Mpc (in natural units this is a placeholder)
    h_c = np.sqrt(Omega) / f
    return h_c

# ── PLOTTING (text-based for server) ────────────────────────────────────────

def print_spectrum_table(f, models, f_c=None):
    """Print ASCII table of spectra."""
    print(f"\n{'='*80}")
    print(f"SGWB Spectrum Comparison — f_c = {f_c:.2e} Hz" if f_c else "SGWB Spectrum Comparison")
    print(f"{'='*80}")
    print(f"{'f [Hz]':>12s} | {'GRU':>12s} | {'Inflation':>12s} | {'Strings':>12s} | {'LISA Sens':>12s}")
    print(f"{'-'*80}")

    for i in range(0, len(f), max(1, len(f)//20)):
        fi = f[i]
        gru = models['gru'][i]
        inf = models['inflation'][i]
        strg = models['cosmic_strings'][i]
        sens = models['lisa_sens'][i]
        print(f"{fi:12.2e} | {gru:12.2e} | {inf:12.2e} | {strg:12.2e} | {sens:12.2e}")
    print(f"{'='*80}")

def find_intersections(f, model, sensitivity):
    """Find where model crosses LISA sensitivity (detectability)."""
    diff = model - sensitivity
    crossings = []
    for i in range(len(diff)-1):
        if diff[i] * diff[i+1] < 0:  # Sign change
            crossings.append(f[i])
    return crossings

# ── MAIN ANALYSIS ───────────────────────────────────────────────────────────

def run_spectrum_analysis(f_screen, f_c, A_gru, mode='plot'):
    """Run full spectrum analysis."""
    print(f"{'='*80}")
    print(f"GRU-LISA SGWB Spectrum Analysis")
    print(f"{'='*80}")
    print(f"A_GRU (derived): {A_gru:.3e}")
    print(f"f_screen: {f_screen:.3f}")
    print(f"f_c (break): {f_c:.2e} Hz")
    print(f"d_s(spine): {D_SPINE:.2f}")
    print(f"d_s(bulk): {D_FULL:.2f}")
    print(f"Mode: {mode}")

    # Frequency grid
    f = np.logspace(np.log10(F_LISA_MIN), np.log10(F_LISA_MAX), 500)

    # GRU spectrum
    Omega_GRU = gru_spectrum(f, A_gru, f_c, f_screen)

    # Standard models
    Omega_inflation = standard_model_spectrum(f, 1e-15, -2.0)
    Omega_strings = standard_model_spectrum(f, 1e-16, -3.0)
    Omega_walls = standard_model_spectrum(f, 1e-14, 1.0)

    # LISA sensitivity
    S_LISA = lisa_sensitivity_curve(f)

    # Store in dict
    models = {
        'gru': Omega_GRU,
        'inflation': Omega_inflation,
        'cosmic_strings': Omega_strings,
        'domain_walls': Omega_walls,
        'lisa_sens': S_LISA,
    }

    # Print table
    print_spectrum_table(f, models, f_c)

    # Find detectability
    crossings = find_intersections(f, Omega_GRU, S_LISA)
    print(f"\n{'='*80}")
    print(f"DETECTABILITY ANALYSIS")
    print(f"{'='*80}")
    if crossings:
        print(f"GRU crosses LISA sensitivity at:")
        for fc in crossings:
            print(f"  f = {fc:.2e} Hz")
        print(f"  Detectable band: [{min(crossings):.2e}, {max(crossings):.2e}] Hz")
    else:
        print(f"GRU does NOT cross LISA sensitivity with current f_screen={f_screen}")
        print(f"  (Expected: needs f_screen calibration from T-scan 3D)")

    # Key frequencies
    print(f"\n  f_c (GRU break): {f_c:.2e} Hz")
    print(f"  LISA peak: {F_LISA_PEAK:.2e} Hz")
    if f_c < F_LISA_MAX and f_c > F_LISA_MIN:
        print(f"  -> Break is WITHIN LISA band!")
    else:
        print(f"  -> Break is OUTSIDE LISA band")

    # Amplitude at peak
    idx_peak = np.argmin(np.abs(f - F_LISA_PEAK))
    print(f"\n  Omega_GRU at LISA peak: {Omega_GRU[idx_peak]:.2e}")
    print(f"  LISA sensitivity at peak: {S_LISA[idx_peak]:.2e}")
    ratio = Omega_GRU[idx_peak] / S_LISA[idx_peak]
    print(f"  SNR (approx): {ratio:.2f}")

    return {
        'f': f.tolist(),
        'Omega_GRU': Omega_GRU.tolist(),
        'Omega_inflation': Omega_inflation.tolist(),
        'Omega_strings': Omega_strings.tolist(),
        'S_LISA': S_LISA.tolist(),
        'f_c': f_c,
        'f_screen': f_screen,
        'A_gru': A_gru,
        'crossings': crossings,
        'detectable': len(crossings) > 0,
        'snr_peak': float(ratio) if not np.isnan(ratio) else None,
    }

def run_sensitivity_sweep(f_screen_values, f_c, A_gru):
    """Sweep f_screen to find detectability threshold."""
    print(f"\n{'='*80}")
    print(f"SENSITIVITY SWEEP — Finding minimum f_screen for detectability")
    print(f"{'='*80}")
    print(f"{'f_screen':>10s} | {'SNR@peak':>12s} | {'Detectable':>12s}")
    print(f"{'-'*80}")

    f = np.logspace(np.log10(F_LISA_MIN), np.log10(F_LISA_MAX), 500)
    S_LISA = lisa_sensitivity_curve(f)
    idx_peak = np.argmin(np.abs(f - F_LISA_PEAK))

    results = []
    for fs in f_screen_values:
        Omega = gru_spectrum(f, A_gru, f_c, fs)
        snr = Omega[idx_peak] / S_LISA[idx_peak]
        det = snr > 1.0
        print(f"{fs:10.3f} | {snr:12.2f} | {'YES' if det else 'NO':>12s}")
        results.append({'f_screen': fs, 'snr': float(snr), 'detectable': det})

    return results

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='GRU-LISA SGWB Spectrum')
    parser.add_argument('--f_screen', type=float, default=0.50, 
                        help='Screening factor (placeholder, calibrate with T-scan 3D)')
    parser.add_argument('--f_c', type=float, default=3e-3,
                        help='Spectral break frequency [Hz]')
    parser.add_argument('--A_gru', type=float, default=A_GRU_DERIVED,
                        help='GRU amplitude (pre-screening)')
    parser.add_argument('--mode', choices=['plot', 'sensitivity', 'compare'],
                        default='plot', help='Analysis mode')
    args = parser.parse_args()

    if args.mode == 'plot':
        result = run_spectrum_analysis(args.f_screen, args.f_c, args.A_gru, 'plot')

    elif args.mode == 'sensitivity':
        f_screen_vals = np.linspace(0.01, 1.0, 20)
        result = run_sensitivity_sweep(f_screen_vals, args.f_c, args.A_gru)

    elif args.mode == 'compare':
        # Compare all standard models + GRU
        f = np.logspace(np.log10(F_LISA_MIN), np.log10(F_LISA_MAX), 500)
        models_to_plot = ['inflation', 'cosmic_strings', 'domain_walls', 'gru']
        print(f"\nComparing {len(models_to_plot)} models...")
        result = run_spectrum_analysis(args.f_screen, args.f_c, args.A_gru, 'compare')

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'mode': args.mode,
        'parameters': {
            'f_screen': args.f_screen,
            'f_c': args.f_c,
            'A_gru': args.A_gru,
            'd_s_spine': D_SPINE,
            'd_s_bulk': D_FULL,
        },
        'results': result,
    }

    with open('GRU_LISA_SGWB_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: GRU_LISA_SGWB_results.json")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
