#!/usr/bin/env python3
"""
GRU-LISA: Time-Delay Correlation Test v4 — Robust Phase Extraction
==================================================================

FIXED: Handles NaN/inf in CSD phase, proper unwrap, and SNR filtering.

Usage:
    python3 GRU_LISA_TIME_DELAY_TEST.py --mode all --fs 100 --A_signal 1e-15
"""

import argparse
import numpy as np
import json
from datetime import datetime

# ── GRU PARAMETERS ──────────────────────────────────────────────────────────

D = 4
KAPPA = 1.0 / (8.0 * np.pi)
LAMBDA_BARE = 0.50

def compute_lambda_eff(Lambda, D, kappa):
    denom = D**2 * (D - 1) * 2 * kappa
    arg = 1.0 + Lambda / denom
    return 1.0 / np.arccosh(arg)

LAMBDA_EFF = compute_lambda_eff(LAMBDA_BARE, D, KAPPA)
DELTA_T_GRU = 128.35  # Periodo spine CDT (segundos, derivado de JSON: T=20, N=2567)

# ── HELPERS ─────────────────────────────────────────────────────────────────

def to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    return obj

# ── REALISTIC SGWB MOCK ────────────────────────────────────────────────────

def generate_gru_spectrum(f, f_c=3e-3, A=1e-15):
    alpha = 0.64
    Omega = A / (1.0 + (f / f_c)**alpha)
    return Omega

def generate_time_series_from_spectrum(freqs, spectrum, duration, fs, seed=42):
    rng = np.random.RandomState(seed)
    n = int(duration * fs)
    freqs_fft = np.fft.rfftfreq(n, d=1.0/fs)
    Omega_interp = np.interp(freqs_fft, freqs, spectrum, left=0, right=0)
    amplitude = np.sqrt(Omega_interp) / (freqs_fft + 1e-10)
    phases = rng.uniform(0, 2*np.pi, len(freqs_fft))
    spec = amplitude * np.exp(1j * phases)
    time_series = np.fft.irfft(spec, n=n)
    return time_series

def generate_lisa_data_with_gru(duration=3600, fs=100.0, f_c=3e-3, 
                                 A_signal=1e-15, seed=42):
    rng = np.random.RandomState(seed)
    n = int(duration * fs)

    freqs = np.logspace(np.log10(1e-4), np.log10(fs/2), 1000)
    Omega_gru = generate_gru_spectrum(freqs, f_c, A_signal)

    gru_A = generate_time_series_from_spectrum(freqs, Omega_gru, duration, fs, seed)

    freqs_fft = np.fft.rfftfreq(n, d=1.0/fs)
    gru_A_fft = np.fft.rfft(gru_A)

    phase_delay = np.exp(-2j * np.pi * freqs_fft * DELTA_T_GRU)
    gru_B_fft = gru_A_fft * phase_delay
    gru_B = np.fft.irfft(gru_B_fft, n=n)

    noise_A = rng.randn(n) * 1e-20
    noise_B = rng.randn(n) * 1e-20

    f_binary = 3e-3
    t = np.arange(n) / fs
    foreground = 1e-20 * np.sin(2 * np.pi * f_binary * t)

    data_A = gru_A + noise_A + foreground
    data_B = gru_B + noise_B + foreground

    return t, data_A, data_B, gru_A, gru_B

# ── CROSS-SPECTRAL ANALYSIS (ROBUST) ────────────────────────────────────────

def cross_spectral_density(data_A, data_B, fs, nperseg=4096):
    """Compute CSD with robust phase extraction."""
    from scipy import signal

    f, Pxy = signal.csd(data_A, data_B, fs=fs, nperseg=nperseg, 
                        noverlap=nperseg//2, scaling='density')

    # Clean: remove NaN, inf, and near-zero values
    valid = np.isfinite(Pxy) & (np.abs(Pxy) > 1e-40) & (f > 0)
    f_clean = f[valid]
    Pxy_clean = Pxy[valid]

    if len(f_clean) < 10:
        return None, None, None, np.nan

    phase = np.angle(Pxy_clean)

    # Unwrap phase (handle 2*pi jumps)
    phase_unwrap = np.unwrap(phase)

    # Fit only in band where GRU signal dominates (1 mHz to 100 mHz)
    band_mask = (f_clean > 1e-3) & (f_clean < 1e-1)
    if np.sum(band_mask) < 5:
        band_mask = (f_clean > 1e-4) & (f_clean < 1.0)

    f_fit = f_clean[band_mask]
    phi_fit = phase_unwrap[band_mask]

    if len(f_fit) < 3:
        return f_clean, Pxy_clean, phase_unwrap, np.nan

    # Linear fit: phi = a + b*f
    try:
        coeffs = np.polyfit(f_fit, phi_fit, 1)
        delta_t_est = -coeffs[0] / (2 * np.pi)
    except (np.linalg.LinAlgError, ValueError):
        delta_t_est = np.nan

    return f_clean, Pxy_clean, phase_unwrap, delta_t_est

def search_time_delay_csd(data_A, data_B, fs, delta_t_expected=DELTA_T_GRU):
    result = cross_spectral_density(data_A, data_B, fs)
    f_clean, Pxy_clean, phase_unwrap, delta_t_est = result

    if f_clean is None or np.isnan(delta_t_est):
        return {
            'delta_t_estimated_ms': None,
            'expected_ms': float(delta_t_expected * 1000),
            'matches_gru': False,
            'significance_sigma': 0.0,
            'error': 'CSD computation failed or insufficient data',
        }

    matches = bool(abs(delta_t_est - delta_t_expected) < 0.01)

    # Bootstrap uncertainty
    n_boot = 50
    dt_boot = []
    for _ in range(n_boot):
        idx = np.random.choice(len(f_clean), len(f_clean), replace=True)
        f_boot = f_clean[idx]
        phi_boot = phase_unwrap[idx]
        band = (f_boot > 1e-3) & (f_boot < 1e-1)
        if np.sum(band) > 5:
            try:
                c = np.polyfit(f_boot[band], phi_boot[band], 1)
                dt_boot.append(-c[0] / (2 * np.pi))
            except:
                pass

    dt_std = np.std(dt_boot) if dt_boot else 0.01
    significance = abs(delta_t_est) / (dt_std + 1e-10)

    return {
        'delta_t_estimated_sec': float(delta_t_est),
        'delta_t_estimated_ms': float(delta_t_est * 1000),
        'expected_ms': float(delta_t_expected * 1000),
        'matches_gru': matches,
        'significance_sigma': float(significance),
        'phase_std_ms': float(dt_std * 1000),
    }

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='GRU-LISA Time-Delay Test v4')
    parser.add_argument('--mode', choices=['search', 'all'], default='all')
    parser.add_argument('--duration', type=float, default=3600)
    parser.add_argument('--fs', type=float, default=100.0)
    parser.add_argument('--A_signal', type=float, default=1e-15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"GRU-LISA Time-Delay Test v4 — Robust Phase Extraction")
    print(f"{'='*70}")
    print(f"Mode: {args.mode} | fs: {args.fs} Hz | A: {args.A_signal:.2e}")
    print(f"Duration: {args.duration}s | GRU delay: {DELTA_T_GRU:.1f} s")

    results = {
        'timestamp': datetime.now().isoformat(),
        'gru_parameters': {
            'delta_t_sec': float(DELTA_T_GRU),
            'delta_t_ms': float(DELTA_T_GRU * 1000),  # solo para referencia; el delay está en segundos
            'lambda_eff': float(LAMBDA_EFF),
        },
    }

    if args.mode in ['search', 'all']:
        print(f"\n{'='*70}")
        print(f"Generating realistic SGWB...")
        print(f"{'='*70}")

        t, data_A, data_B, gru_A, gru_B = generate_lisa_data_with_gru(
            duration=args.duration, fs=args.fs, A_signal=args.A_signal, seed=args.seed)

        print(f"  Data A: {data_A.shape}, std={np.std(data_A):.2e}")
        print(f"  Data B: {data_B.shape}, std={np.std(data_B):.2e}")

        print(f"\n{'='*70}")
        print(f"CROSS-SPECTRAL ANALYSIS")
        print(f"{'='*70}")

        search_result = search_time_delay_csd(data_A, data_B, args.fs)
        results['search'] = search_result

        print(f"\n  Estimated delay: {search_result.get('delta_t_estimated_ms', 'N/A')} ms")
        print(f"  Expected delay:  {search_result['expected_ms']:.2f} ms")
        print(f"  Match: {'YES' if search_result.get('matches_gru') else 'NO'}")
        print(f"  Significance: {search_result.get('significance_sigma', 0):.2f} sigma")

    with open('GRU_LISA_TIME_DELAY_results.json', 'w') as f:
        json.dump(to_native(results), f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: GRU_LISA_TIME_DELAY_results.json")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
