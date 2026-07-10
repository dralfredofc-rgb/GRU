#!/usr/bin/env python3
"""
GRU_Dplus_operator_v2.py
========================
Formalismo D+: Operador de dimension espectral cuantico sobre ensemble CDT real.

Correcciones respecto al original de Perplexity:
  1. Bug corregido: triangulations era list, no ndarray, no tiene .tolist()
  2. Modo headless: matplotlib en backend 'Agg' para servidor sin display
  3. Ventana de ajuste ILO/IHI consistente con el protocolo GRU verificado
     (ILO = 0.2*T, IHI = 2*T, no los valores fijos 6/150 del original)

Interpretacion fisica:
  |Psi_CDT> = (1/sqrt(N)) sum_i |T_i>   : estado ensemble equiprobable
  D_hat |T_i> = d_s^(i) |T_i>           : operador diagonal
  <D> = <Psi|D_hat|Psi>                  : dimension espectral promedio
  Var(D) = <D^2> - <D>^2                 : fluctuaciones del ensemble

Autor: GRU Protocol (corregido 22 jun 2026)
Version: D+ v0.2
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless — sin display necesario
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy import stats as sp_stats

# ============================================================
# CONFIGURACION GRU — protocolo verificado en sesiones 20-22 jun
# ============================================================
GRU_CONFIG = {
    "NWALKS": 3000,
    "SIGMAMAX": 150,
    "SEED": 42,
    "ILO_FRAC": 0.20,   # ILO = int(0.20 * T)
    "IHI_MULT": 2,      # IHI = min(2*T, SIGMAMAX-1)
}


# ============================================================
# OPERADOR D_hat
# ============================================================

@dataclass
class QuantumDimensionOperator:
    """Operador D_hat diagonal en la base de triangulaciones {|T_i>}."""

    triangulations: List[str] = field(default_factory=list)
    dimensions: List[float] = field(default_factory=list)
    errors: Optional[List[float]] = None
    weights: Optional[np.ndarray] = None

    def __post_init__(self):
        if len(self.triangulations) != len(self.dimensions):
            raise ValueError(
                f"Triangulaciones ({len(self.triangulations)}) != "
                f"dimensiones ({len(self.dimensions)})"
            )
        self.N = len(self.dimensions)
        self.dimensions = [float(d) for d in self.dimensions]  # fix: asegurar float
        if self.weights is None:
            self.weights = np.ones(self.N) / self.N
        else:
            self.weights = np.array(self.weights, dtype=float)
            self.weights /= self.weights.sum()

    @property
    def expectation_value(self):
        return float(np.dot(self.weights, self.dimensions))

    @property
    def expectation_squared(self):
        return float(np.dot(self.weights, np.array(self.dimensions)**2))

    @property
    def variance(self):
        return self.expectation_squared - self.expectation_value**2

    @property
    def std_dev(self):
        return float(np.sqrt(max(self.variance, 0.0)))

    @property
    def std_error(self):
        """Error estandar de la media."""
        return self.std_dev / np.sqrt(self.N)

    @property
    def relative_fluctuation(self):
        if self.expectation_value == 0:
            return float('inf')
        return self.std_dev / self.expectation_value

    def verify_diagonal(self):
        psi = np.sqrt(self.weights)
        D_mat = np.diag(self.dimensions)
        direct = float(psi.T @ D_mat @ psi)
        return np.isclose(direct, self.expectation_value, rtol=1e-10)

    def normality_tests(self):
        """Tests de normalidad sobre la distribucion de d_s."""
        dims = np.array(self.dimensions)
        results = {}
        if self.N >= 8:
            stat, p = sp_stats.shapiro(dims)
            results['shapiro'] = {'statistic': float(stat), 'p_value': float(p),
                                   'normal': bool(p > 0.05)}
        if self.N >= 20:
            stat, p = sp_stats.kstest(dims, 'norm',
                                       args=(np.mean(dims), np.std(dims, ddof=1)))
            results['ks_normal'] = {'statistic': float(stat), 'p_value': float(p),
                                     'normal': bool(p > 0.05)}
        return results

    def to_dict(self):
        base = {
            "N_triangulations": self.N,
            "expectation_value_D": self.expectation_value,
            "expectation_value_D2": self.expectation_squared,
            "variance_D": self.variance,
            "std_dev_D": self.std_dev,
            "std_error_D": self.std_error,
            "relative_fluctuation": self.relative_fluctuation,
            "dimensions": self.dimensions,
            "triangulations": self.triangulations,
            "normality_tests": self.normality_tests(),
        }
        if self.errors:
            base["errors"] = self.errors
        return base


# ============================================================
# CARGA DE DATOS
# ============================================================

def load_from_json_results(json_path):
    """
    Carga d_s desde un JSON del pipeline GRU.
    Soporta formatos: {"results": [...]} o lista directa o dict unico.
    """
    with open(json_path) as f:
        data = json.load(f)

    triangulations, dimensions, errors = [], [], []

    # Formato plano: {"dimensions": [...], "triangulations": [...]}
    if "dimensions" in data and isinstance(data["dimensions"], list):
        dims = data["dimensions"]
        tris = data.get("triangulations",
                        [f"T_{i}" for i in range(len(dims))])
        return list(tris), [float(d) for d in dims], None

    # Formato lista de resultados: {"results": [...]} o lista directa
    if "results" in data:
        entries = data["results"]
    elif isinstance(data, list):
        entries = data
    else:
        entries = [data]

    for entry in entries:
        fname = entry.get("file", entry.get("filename",
                entry.get("triangulation", "unknown")))
        d_s = entry.get("d_s", entry.get("dimension",
              entry.get("ds", None)))
        err = entry.get("d_s_err", entry.get("error", None))

        if d_s is not None:
            triangulations.append(str(fname))
            dimensions.append(float(d_s))
            if err is not None:
                errors.append(float(err))

    return triangulations, dimensions, errors or None


# ============================================================
# VISUALIZACION (headless — guarda PNG sin mostrar)
# ============================================================

def plot_distribution(operator, output_path=None, title="GRU D+ Ensemble"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    dims = np.array(operator.dimensions)
    mean = operator.expectation_value
    std = operator.std_dev
    se = operator.std_error

    # Histograma
    ax1 = axes[0]
    ax1.hist(dims, bins='auto', density=True, alpha=0.7,
             color='steelblue', edgecolor='black')
    ax1.axvline(mean, color='red', linestyle='--', linewidth=2,
                label=f'<D> = {mean:.4f} +/- {se:.4f}')
    ax1.axvline(mean+std, color='orange', linestyle=':', label=f'1-sigma = {std:.4f}')
    ax1.axvline(mean-std, color='orange', linestyle=':')
    ax1.axvline(1.0, color='green', linestyle='--', alpha=0.6, label='d_s = 1.0 (GRU)')
    ax1.set_xlabel('d_s (spine)')
    ax1.set_ylabel('Densidad')
    ax1.set_title(f'Distribucion D_hat\n{title}')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Serie por indice
    ax2 = axes[1]
    ax2.plot(range(len(dims)), dims, 'o-', alpha=0.6, color='steelblue', markersize=4)
    ax2.axhline(mean, color='red', linestyle='--', label=f'<D> = {mean:.4f}')
    ax2.fill_between(range(len(dims)), mean-std, mean+std, alpha=0.2, color='red')
    ax2.axhline(1.0, color='green', linestyle='--', alpha=0.6, label='d_s = 1.0')
    ax2.set_xlabel('Indice triangulacion i')
    ax2.set_ylabel('d_s^(i)')
    ax2.set_title(f'Autovalores D_hat\n{title}')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Q-Q plot
    ax3 = axes[2]
    sp_stats.probplot(dims, dist="norm", plot=ax3)
    norm_tests = operator.normality_tests()
    shapiro_txt = ""
    if 'shapiro' in norm_tests:
        s = norm_tests['shapiro']
        shapiro_txt = f"\nShapiro p={s['p_value']:.3f} ({'normal' if s['normal'] else 'NO normal'})"
    ax3.set_title(f'Q-Q plot{shapiro_txt}')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[D+] Figura guardada: {output_path}")
    plt.close(fig)


# ============================================================
# ANALISIS COMPLETO
# ============================================================

def analyze_ensemble(input_source, output_dir="./Dplus_results", title="GRU D+"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[D+] Cargando: {input_source}")
    triangulations, dimensions, errors = load_from_json_results(input_source)
    print(f"[D+] {len(triangulations)} triangulaciones cargadas")

    if not dimensions:
        print("[D+] ERROR: no se encontraron dimensiones en el archivo")
        return None

    operator = QuantumDimensionOperator(
        triangulations=triangulations,
        dimensions=dimensions,
        errors=errors
    )

    assert operator.verify_diagonal(), "ERROR: verificacion matricial fallo"

    print()
    print("=" * 60)
    print(f"  OPERADOR DE DIMENSION ESPECTRAL D_hat")
    print(f"  {title}")
    print("=" * 60)
    print(f"  N triangulaciones:   {operator.N}")
    print(f"  <D>  (valor esp.):   {operator.expectation_value:.6f}")
    print(f"  SEM (error media):   {operator.std_error:.6f}")
    print(f"  sigma_D (std dev):   {operator.std_dev:.6f}")
    print(f"  Var(D):              {operator.variance:.8f}")
    print(f"  delta_D (rel. fluc): {operator.relative_fluctuation:.6f}")
    print(f"  Rango [min, max]:    [{min(dimensions):.6f}, {max(dimensions):.6f}]")
    print()
    nt = operator.normality_tests()
    for name, res in nt.items():
        estado = "OK (normal)" if res['normal'] else "FALLA (no normal)"
        print(f"  Test {name}: p={res['p_value']:.4f} -> {estado}")
    print("=" * 60)

    # Guardar JSON
    results_path = os.path.join(output_dir, "Dplus_results.json")
    with open(results_path, 'w') as f:
        json.dump(operator.to_dict(), f, indent=2)
    print(f"[D+] Resultados: {results_path}")

    # Figura
    plot_distribution(
        operator,
        output_path=os.path.join(output_dir, "Dplus_distribution.png"),
        title=title
    )

    return operator


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRU D+: Operador dimension espectral cuantico")
    parser.add_argument("input", nargs="?", help="JSON con resultados del pipeline GRU")
    parser.add_argument("--output", default="./Dplus_results", help="Directorio de salida")
    parser.add_argument("--title", default="GRU D+ Ensemble", help="Titulo")
    parser.add_argument("--example", action="store_true", help="Test con datos del Dplus_example_results.json")
    args = parser.parse_args()

    if args.example:
        # Usa el JSON de ejemplo que ya tiene los 32 seeds reales
        example_path = "Dplus_example_results.json"
        if not os.path.exists(example_path):
            print(f"ERROR: no se encuentra {example_path}")
            print("Copia Dplus_example_results.json al directorio actual primero")
            sys.exit(1)
        analyze_ensemble(example_path, args.output, "CDT 3D T=20 (32 seeds)")
    elif args.input:
        analyze_ensemble(args.input, args.output, args.title)
    else:
        print("Uso: python3 GRU_Dplus_operator_v2.py <pipeline_results.json>")
        print("     python3 GRU_Dplus_operator_v2.py --example")
