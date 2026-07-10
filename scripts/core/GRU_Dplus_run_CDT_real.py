#!/usr/bin/env python3
"""
GRU_Dplus_run_CDT_real.py
=========================
Calcula el operador D+ directamente sobre los 32 JSON de CDT 3D T=20
ya exportados en el servidor. Calcula d_s con el protocolo GRU verificado
(heat-kernel + fit log-log, vertex_times reales, NO BFS) en vez de
leer d_s de un archivo externo.

USO:
  python3 GRU_Dplus_run_CDT_real.py
  python3 GRU_Dplus_run_CDT_real.py --dir /root/3d-cdt/example --pattern "GRU_CDT3D_T20_s*.json"
  python3 GRU_Dplus_run_CDT_real.py --dir /root/3d-cdt/example --pattern "GRU_CDT3D_T*.json"

Tambien puedes incluir 2D:
  python3 GRU_Dplus_run_CDT_real.py \\
    --dir /root/2d-cdt/example --pattern "GRU_CDT2D_T20_s*.json" \\
    --output ./Dplus_2D

Requisitos:
  - Los JSON deben tener 'vertex_times' y 'edges' (formato del pipeline GRU)
  - scipy, numpy, matplotlib instalados

Autor: GRU Protocol (22 jun 2026)
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats as sp_stats

# ============================================================
# PROTOCOLO GRU VERIFICADO (identico a GRU_C1_validacion_CDT_real.py)
# ============================================================
NWALKS = 3000
SIGMAMAX = 150
SEED_RW = 42   # seed para las caminatas aleatorias


def get_spine(edges, vt, T):
    """Spine colapsado — identico al pipeline real verificado."""
    shell_edges_found = set()
    for u, v in edges:
        tu, tv = int(vt[u]), int(vt[v])
        if abs(tu - tv) == 1:
            shell_edges_found.add(min(tu, tv))
    spine_adj = [[] for _ in range(T)]
    for t in range(T - 1):
        if t in shell_edges_found:
            spine_adj[t].append(t + 1)
            spine_adj[t + 1].append(t)
    # Cierre S^1 (correccion topologica verificada)
    spine_adj[T - 1].append(0)
    spine_adj[0].append(T - 1)
    return spine_adj


def heat_kernel(spine_adj, T, nw, sm, seed):
    """Heat kernel sobre el spine. Identico al protocolo verificado."""
    rng = np.random.default_rng(seed)
    P = np.zeros(sm)
    for _ in range(nw):
        cur = 0
        for step in range(sm):
            if cur == 0:
                P[step] += 1.0
            nb = spine_adj[cur]
            if not nb:
                break
            cur = nb[rng.integers(len(nb))]
    return P / nw


def fit_ds(P, T, sigma_max):
    """Fit ds = 2*alpha via P(sigma) ~ A * sigma^(-alpha). Protocolo GRU."""
    ilo = max(3, int(0.2 * T))
    ihi = min(2 * T, sigma_max - 1)
    sa = np.arange(1, sigma_max + 1, dtype=float)
    s_fit, p_fit = sa[ilo:ihi], P[ilo:ihi]
    mask = p_fit > 1e-10
    if mask.sum() < 4:
        return None, None
    try:
        po, pc = curve_fit(
            lambda s, A, a: A * s**(-a),
            s_fit[mask], p_fit[mask],
            p0=(p_fit[mask][0], 0.5),
            maxfev=5000,
            bounds=([0, 0.01], [np.inf, 3.0])
        )
        return 2 * po[1], 2 * np.sqrt(np.diag(pc)[1])
    except Exception:
        return None, None


def compute_ds_from_json(json_path):
    """
    Carga un JSON del pipeline GRU y calcula d_s con el protocolo verificado.
    Usa vertex_times reales (NO BFS).
    """
    with open(json_path) as f:
        d = json.load(f)

    N = d['N']
    T = d['T']
    edges = d['edges']
    vt = d['vertex_times']

    spine_adj = get_spine(edges, vt, T)
    P = heat_kernel(spine_adj, T, NWALKS, SIGMAMAX, SEED_RW)
    ds, err = fit_ds(P, T, SIGMAMAX)
    return ds, err, T, N


# ============================================================
# OPERADOR D+ (version ligera, sin dependencia de GRU_Dplus_operator_v2)
# ============================================================

def compute_Dplus(triangulations, dimensions, errors=None):
    N = len(dimensions)
    dims = np.array(dimensions)
    w = np.ones(N) / N
    mean_D = float(np.dot(w, dims))
    mean_D2 = float(np.dot(w, dims**2))
    var_D = mean_D2 - mean_D**2
    std_D = float(np.sqrt(max(var_D, 0.0)))
    sem_D = std_D / np.sqrt(N)
    rel_fluc = std_D / mean_D if mean_D != 0 else float('inf')

    # Tests de normalidad
    norm = {}
    if N >= 8:
        stat, p = sp_stats.shapiro(dims)
        norm['shapiro'] = {'stat': float(stat), 'p': float(p), 'normal': p > 0.05}
    if N >= 20:
        stat, p = sp_stats.kstest(dims, 'norm', args=(np.mean(dims), np.std(dims, ddof=1)))
        norm['ks'] = {'stat': float(stat), 'p': float(p), 'normal': p > 0.05}

    return {
        'N': N,
        'mean_D': mean_D,
        'sem_D': sem_D,
        'std_D': std_D,
        'var_D': float(var_D),
        'rel_fluc': float(rel_fluc),
        'min_D': float(dims.min()),
        'max_D': float(dims.max()),
        'dimensions': dimensions,
        'triangulations': triangulations,
        'errors': errors,
        'normality': norm,
    }


# ============================================================
# VISUALIZACION
# ============================================================

def plot_Dplus(result, output_path=None, title="GRU D+ Ensemble CDT Real"):
    dims = np.array(result['dimensions'])
    mean = result['mean_D']
    std = result['std_D']
    sem = result['sem_D']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histograma
    ax = axes[0]
    ax.hist(dims, bins='auto', density=True, alpha=0.7,
            color='steelblue', edgecolor='black')
    ax.axvline(mean, color='red', linestyle='--', linewidth=2,
               label=f'<D> = {mean:.4f} +/- {sem:.4f} (SEM)')
    ax.axvline(mean + std, color='orange', linestyle=':', label=f'1-sigma = {std:.4f}')
    ax.axvline(mean - std, color='orange', linestyle=':')
    ax.axvline(1.0, color='green', linestyle='--', alpha=0.7, label='d_s = 1.0 (GRU)')
    ax.set_xlabel('d_s (spine, protocolo GRU verificado)')
    ax.set_ylabel('Densidad')
    ax.set_title(f'Distribucion D_hat\n{title}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Serie por indice
    ax = axes[1]
    ax.plot(range(len(dims)), dims, 'o-', alpha=0.7, color='steelblue', markersize=4)
    ax.axhline(mean, color='red', linestyle='--', label=f'<D> = {mean:.4f}')
    ax.fill_between(range(len(dims)), mean - std, mean + std,
                    alpha=0.2, color='red', label='1-sigma')
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.7, label='d_s = 1.0')
    if result.get('errors'):
        ax.errorbar(range(len(dims)), dims,
                    yerr=result['errors'], fmt='none', color='gray', alpha=0.4)
    ax.set_xlabel('Indice seed i')
    ax.set_ylabel('d_s^(i)')
    ax.set_title(f'Autovalores D_hat\n{title}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Q-Q plot
    ax = axes[2]
    sp_stats.probplot(dims, dist="norm", plot=ax)
    norm = result.get('normality', {})
    txt = ""
    if 'shapiro' in norm:
        s = norm['shapiro']
        txt += f"\nShapiro p={s['p']:.3f} ({'normal' if s['normal'] else 'NO normal'})"
    ax.set_title(f'Q-Q plot (normalidad){txt}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[D+] Figura: {output_path}")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GRU D+: Calcula operador cuantico sobre JSON CDT reales"
    )
    parser.add_argument(
        "--dir", default="/root/3d-cdt/example",
        help="Directorio con JSON del pipeline GRU (default: /root/3d-cdt/example)"
    )
    parser.add_argument(
        "--pattern", default="GRU_CDT3D_T20_s*.json",
        help="Patron glob de archivos JSON (default: GRU_CDT3D_T20_s*.json)"
    )
    parser.add_argument(
        "--output", default="./Dplus_CDT_real",
        help="Directorio de salida (default: ./Dplus_CDT_real)"
    )
    parser.add_argument(
        "--title", default="CDT 3D T=20 (seeds reales)",
        help="Titulo para figuras"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Buscar archivos
    patron = os.path.join(args.dir, args.pattern)
    files = sorted(glob.glob(patron))
    if not files:
        print(f"ERROR: no se encontraron archivos con patron: {patron}")
        sys.exit(1)

    print(f"[D+] Encontrados {len(files)} archivos en {args.dir}")
    print(f"[D+] Calculando d_s con protocolo GRU verificado (NWALKS={NWALKS})...")
    print()

    triangulations, dimensions, errors = [], [], []
    failed = []

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        ds, err, T, N = compute_ds_from_json(fpath)
        if ds is None:
            print(f"  [{i+1:2d}/{len(files)}] {fname}: FIT ERROR — saltando")
            failed.append(fname)
            continue
        triangulations.append(fname)
        dimensions.append(ds)
        if err is not None:
            errors.append(err)
        print(f"  [{i+1:2d}/{len(files)}] {fname}: "
              f"N={N}, T={T}, ds={ds:.4f} +/- {err:.4f}")

    print()
    if not dimensions:
        print("ERROR: ningun archivo pudo procesarse correctamente")
        sys.exit(1)

    # Calcular operador D+
    result = compute_Dplus(
        triangulations, dimensions,
        errors=errors if errors else None
    )

    # Reporte
    print("=" * 65)
    print("  OPERADOR DE DIMENSION ESPECTRAL D_hat — RESULTADO FINAL")
    print(f"  {args.title}")
    print("=" * 65)
    print(f"  N triangulaciones:        {result['N']}")
    print(f"  <D_hat> (valor esperado): {result['mean_D']:.6f}")
    print(f"  SEM (error de la media):  {result['sem_D']:.6f}")
    print(f"  sigma_D (std del ens.):   {result['std_D']:.6f}")
    print(f"  Var(D):                   {result['var_D']:.8f}")
    print(f"  delta_D (rel. fluct.):    {result['rel_fluc']:.6f} ({result['rel_fluc']*100:.2f}%)")
    print(f"  Rango [min, max]:         [{result['min_D']:.6f}, {result['max_D']:.6f}]")
    print()
    print("  Interpretacion fisica:")
    print(f"  <D_hat> = {result['mean_D']:.4f} +/- {result['sem_D']:.4f}")
    print(f"  Consistente con d_s(spine) ~ 1? "
          f"{'SI' if abs(result['mean_D'] - 1.0) < 3 * result['sem_D'] else 'NO'}")
    print(f"  (|<D> - 1.0| = {abs(result['mean_D']-1.0):.4f}, "
          f"3*SEM = {3*result['sem_D']:.4f})")
    print()
    for nombre, res in result['normality'].items():
        estado = "normal" if res['normal'] else "NO normal"
        print(f"  Test {nombre}: p={res['p']:.4f} -> distribucion {estado}")
    if failed:
        print(f"\n  Archivos fallidos ({len(failed)}): {', '.join(failed)}")
    print("=" * 65)

    # Guardar
    results_path = os.path.join(args.output, "Dplus_CDT_real_results.json")
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[D+] Resultados JSON: {results_path}")

    plot_Dplus(
        result,
        output_path=os.path.join(args.output, "Dplus_CDT_real_distribution.png"),
        title=args.title
    )
    print("[D+] Listo.")


if __name__ == "__main__":
    main()
