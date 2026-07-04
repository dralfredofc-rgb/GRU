#!/usr/bin/env python3
"""
GRU_P_ANISOTROPY_v2_5.py — P_ANISOTROPY v2.5

Calcula la variacion cuadrupolar de la amplitud MDR con la direccion del cielo
desde la geometria del spine GRU (Origen radial unico).

Formula: A_GRU(n) = A_GRU * [1 + epsilon_A * P_2(cos theta)]

donde:
  epsilon_A = 0.0412 (4.12%) — derivado de pipeline A.21 (memoria 55)
  P_2(x) = (3x^2 - 1)/2 — polinomio de Legendre cuadrupolar
  theta = angulo entre direccion de la fuente y Origen radial

Inputs: JSON con epsilon_A, A_GRU, n_GRU
Output: JSON con amplitud angular, mapa de dephasing, firma GRU

DOI v2.4: 10.5281/zenodo.20939080
"""

import numpy as np
import json
import os
import argparse

OUTPUT_DIR = os.environ.get("GRU_OUTPUT_DIR", "/root")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def p2(x):
    """Polinomio de Legendre cuadrupolar P_2(cos theta)."""
    return 0.5 * (3.0 * x**2 - 1.0)

def compute_anisotropy(params, n_theta=100, n_phi=100):
    """Calcula mapa de amplitud MDR anisotropica sobre esfera del cielo."""
    epsilon_A = params['epsilon_A']
    A_GRU = params['A_GRU']
    n_GRU = params['n_GRU']

    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi)

    # Variacion cuadrupolar
    cos_theta = np.cos(THETA)
    P2_vals = p2(cos_theta)
    A_map = A_GRU * (1.0 + epsilon_A * P2_vals)

    # Firma unica: coeficiente a_20 (j=2, m=0)
    # Integral sobre esfera: a_20 = int(A * P_2 * sin(theta) dtheta dphi) / norm
    dtheta = np.pi / (n_theta - 1)
    dphi = 2 * np.pi / (n_phi - 1)
    integrand = A_map * P2_vals * np.sin(THETA)
    a_20 = np.sum(integrand) * dtheta * dphi / (4 * np.pi)

    return {
        'theta_grid': theta.tolist(),
        'phi_grid': phi.tolist(),
        'A_map_shape': [n_phi, n_theta],
        'A_max': float(np.max(A_map)),
        'A_min': float(np.min(A_map)),
        'A_mean': float(np.mean(A_map)),
        'delta_A_over_A': float((np.max(A_map) - np.min(A_map)) / A_GRU),
        'a_20_coefficient': float(a_20),
        'epsilon_A': epsilon_A,
        'A_GRU': A_GRU,
        'n_GRU': n_GRU,
        'f_spine': params.get('f_spine', 7.877e-3),
        'signature': 'j=2, m=0 quadrupolar anisotropy — GRU unique'
    }

def main():
    parser = argparse.ArgumentParser(description='GRU P_ANISOTROPY v2.5')
    parser.add_argument('--input', default='GRU_QISKIT_master_output_REAL.json',
                        help='JSON con epsilon_A, A_GRU, n_GRU')
    parser.add_argument('--n_theta', type=int, default=100)
    parser.add_argument('--n_phi', type=int, default=100)
    args = parser.parse_args()

    # Valores validados GRU v2.4 (memoria 55, 57)
    DEFAULT_PARAMS = {
        'epsilon_A': 0.0412,        # Pipeline A.21
        'A_GRU': 4.235e-8,          # Valor LISA v2.4 (memoria 57)
        'n_GRU': 0.0564,            # gamma = 2*ds_spine - 2
        'f_spine': 7.877e-3         # mHz, memoria 57
    }

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"[WARNING] {input_path} no encontrado. Usando valores v2.4 validados.")
        params = DEFAULT_PARAMS.copy()
    else:
        with open(input_path, 'r') as f:
            data = json.load(f)
        params = {
            'epsilon_A': data.get('epsilon_A', DEFAULT_PARAMS['epsilon_A']),
            'A_GRU': data.get('A_GRU', DEFAULT_PARAMS['A_GRU']),
            'n_GRU': data.get('n_GRU', DEFAULT_PARAMS['n_GRU']),
            'f_spine': data.get('f_spine', DEFAULT_PARAMS['f_spine'])
        }

    print("="*60)
    print("GRU P_ANISOTROPY v2.5 — Firma cuadrupolar del Origen radial")
    print("="*60)
    print(f"epsilon_A = {params['epsilon_A']:.4f}")
    print(f"A_GRU = {params['A_GRU']:.4e}")
    print(f"n_GRU = {params['n_GRU']:.4f}")
    print(f"f_spine = {params['f_spine']:.4e} mHz")

    result = compute_anisotropy(params, n_theta=args.n_theta, n_phi=args.n_phi)

    print(f"\n[RESULTADOS]")
    print(f"A_max = {result['A_max']:.4e}")
    print(f"A_min = {result['A_min']:.4e}")
    print(f"Delta A / A = {result['delta_A_over_A']:.4f} ({result['delta_A_over_A']*100:.2f}%)")
    print(f"a_20 (coef. cuadrupolar) = {result['a_20_coefficient']:.4e}")
    print(f"Firma: {result['signature']}")

    # Comparacion con prediccion teorica
    delta_A_pred = 2 * params['epsilon_A']  # P_2 va de -1/2 a +1
    print(f"\n[VALIDACION]")
    print(f"Delta A/A teorico = {delta_A_pred:.4f} ({delta_A_pred*100:.2f}%)")
    print(f"Delta A/A calculado = {result['delta_A_over_A']:.4f}")
    if abs(result['delta_A_over_A'] - delta_A_pred) < 0.01:
        print("Consistente con epsilon_A = 4.12%")
    else:
        print("Desviacion de prediccion teorica")

    # Guardar
    out_path = os.path.join(OUTPUT_DIR, "GRU_P_ANISOTROPY_v2_5.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[OUTPUT] Guardado en {out_path}")
    print(f"[STATUS] P_ANISOTROPY v2.5 OK")

if __name__ == "__main__":
    main()
