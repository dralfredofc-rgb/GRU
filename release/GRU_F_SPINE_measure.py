#!/usr/bin/env python3
"""
GRU-LISA F_SPINE_measure: Medir f_spine en geometrías CDT reales
=================================================================
Alfredo Flores Cornejo — GRU v2.x — clouding.io

OBJETIVO: Medir f_spine = N_spine / N_total en tus JSON CDT reales.
El spine GRU es el ciclo S¹ de T nodos representantes (uno por shell),
extraído usando vertex_times reales (protocolo GRU — NUNCA BFS sobre
grafo completo, que da error sistemático ~1.4 en 3D).

FORMATO JSON GRU REAL (Brunekreef/clouding.io):
  {
    "N": 2567,
    "T": 20,
    "vertex_times": [0,0,0,1,1,...,19,19],
    "edges": [[0,1],[0,5],...]
  }

DEFINICIÓN DE SPINE GRU:
  - Para cada shell t in [0,T-1]: representante = primer nodo con
    vertex_times[nodo] == t que tenga al menos una arista temporal
    (|Δt|=1) hacia el siguiente shell.
  - N_spine = T (siempre, por construcción del protocolo GRU)
  - f_spine = N_spine / N_total = T / N

NOTA SOBRE f_spine:
  En CDT con T=20 y N≈2500 → f_spine = 20/2500 = 8×10⁻³
  La ventana LISA requiere f_eff ∈ [1.95e-9, 1.78e-6]
  → f_spine_bruto = 8 mHz >> ventana → se necesita screening o f_eff << f_spine

USO: python3 GRU_F_SPINE_measure.py /root/*.json
"""
import json, sys, glob, os
from collections import defaultdict
from statistics import mean, pstdev

# Ventana LISA (de P1.5)
F_MIN_WINDOW = 1.95e-9
F_MAX_WINDOW = 1.78e-6


def load_gru_json(path):
    """Carga JSON formato GRU (vertex_times + edges)."""
    with open(path) as f:
        d = json.load(f)
    N  = d.get('N') or len(d.get('vertex_times', []))
    T  = d.get('T', 20)
    vt = d.get('vertex_times', [])
    edges = d.get('edges', [])
    if not vt:
        raise ValueError(f"Sin vertex_times en {path}")
    return N, T, vt, edges


def extract_spine_info(N, T, vt, edges):
    """
    Extrae información del spine GRU.
    
    Spine = T nodos representantes (uno por shell).
    Protocolo: primer nodo de cada shell que tiene conexión
    temporal real (arista con |Δt|=1) al siguiente shell.
    
    Devuelve:
      shells: dict t -> [nodos en shell t]
      spine_reps: lista de T representantes
      temporal_edges: aristas con |Δt|=1
    """
    # Agrupar nodos por shell
    shells = defaultdict(list)
    for node_id, t in enumerate(vt):
        shells[t].append(node_id)
    
    # Clasificar aristas
    temporal = []  # |Δt|=1
    spatial  = []  # |Δt|=0
    periodic = []  # |Δt|>1
    for u, v in edges:
        dt = abs(vt[u] - vt[v])
        if dt == 0:   spatial.append((u,v))
        elif dt == 1: temporal.append((u,v))
        else:          periodic.append((u,v))
    
    # Representante de cada shell: primer nodo con conexión temporal
    # (protocolo GRU real: no BFS arbitrario)
    temporal_neighbors = defaultdict(set)
    for u, v in temporal:
        temporal_neighbors[u].add(v)
        temporal_neighbors[v].add(u)
    
    spine_reps = []
    for t in range(T):
        shell_nodes = shells[t]
        # Preferir nodo con conexiones temporales al shell siguiente
        rep = None
        for node in shell_nodes:
            if any(vt[nb] == t+1 or vt[nb] == t-1 
                   for nb in temporal_neighbors[node]):
                rep = node
                break
        if rep is None and shell_nodes:
            rep = shell_nodes[0]  # fallback
        if rep is not None:
            spine_reps.append(rep)
    
    return shells, spine_reps, temporal, spatial, periodic


def process_file(path):
    N, T, vt, edges = load_gru_json(path)
    shells, spine_reps, temporal, spatial, periodic = extract_spine_info(N, T, vt, edges)
    
    N_spine = len(spine_reps)  # = T por construcción
    f_spine = N_spine / N if N > 0 else 0.0
    
    # Grado medio del spine (aristas temporales por nodo representante)
    spine_set = set(spine_reps)
    spine_temporal = sum(1 for u,v in temporal 
                        if u in spine_set or v in spine_set)
    
    return {
        'file':         os.path.basename(path),
        'N':            N,
        'T':            T,
        'N_spine':      N_spine,
        'f_spine':      f_spine,
        'n_temporal':   len(temporal),
        'n_spatial':    len(spatial),
        'n_periodic':   len(periodic),
        'n_edges_total':len(edges),
        'spine_temporal_edges': spine_temporal,
    }


def main(patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    
    if not files:
        print("No se encontraron archivos. Uso:")
        print("  python3 GRU_F_SPINE_measure.py /root/*.json")
        sys.exit(1)
    
    print("="*70)
    print("GRU-LISA F_SPINE: MEDICIÓN f_spine EN GEOMETRÍAS CDT REALES")
    print("="*70)
    print(f"Archivos: {len(files)}\n")
    
    results = []
    for path in files:
        try:
            r = process_file(path)
            results.append(r)
            print(f"[{r['file']}]")
            print(f"  N={r['N']}, T={r['T']}")
            print(f"  N_spine={r['N_spine']}  f_spine={r['f_spine']:.4e}")
            print(f"  Aristas: total={r['n_edges_total']}, "
                  f"temporal={r['n_temporal']}, "
                  f"spatial={r['n_spatial']}, "
                  f"periodic={r['n_periodic']}")
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
    
    if not results:
        print("No se procesaron geometrías válidas.")
        sys.exit(1)
    
    # Estadísticos
    f_vals = [r['f_spine'] for r in results]
    f_mean = mean(f_vals)
    f_std  = pstdev(f_vals) if len(f_vals) > 1 else 0.0
    
    print(f"\n{'─'*60}")
    print("RESUMEN GLOBAL:")
    print(f"  Geometrías procesadas: {len(results)}")
    print(f"  f_spine = T/N:")
    print(f"    mean = {f_mean:.4e}")
    print(f"    min  = {min(f_vals):.4e}")
    print(f"    max  = {max(f_vals):.4e}")
    print(f"    std  = {f_std:.4e}")
    
    print(f"\nVENTANA LISA (P1.5): f_eff ∈ [{F_MIN_WINDOW:.2e}, {F_MAX_WINDOW:.2e}]")
    print(f"  f_spine_mean = {f_mean:.4e}")
    
    ratio_to_max = f_mean / F_MAX_WINDOW
    ratio_to_min = f_mean / F_MIN_WINDOW
    
    if F_MIN_WINDOW <= f_mean <= F_MAX_WINDOW:
        print(f"  ✅ DENTRO DE LA VENTANA")
    elif f_mean > F_MAX_WINDOW:
        print(f"  ❌ FUERA (demasiado alto en {ratio_to_max:.2e}×)")
        print(f"     → Se necesita screening: f_eff/f_spine ~ {F_MAX_WINDOW/f_mean:.2e}")
    else:
        print(f"  ❌ FUERA (demasiado bajo en {1/ratio_to_min:.2e}×)")
    
    # NOTA FÍSICA IMPORTANTE
    print(f"\nNOTA: El spine GRU siempre tiene N_spine = T nodos.")
    print(f"  Con T=20, N≈2500: f_spine = 20/2500 = 8 mHz")
    print(f"  La ventana LISA requiere f_eff ~ 10⁻⁶ a 10⁻⁹ mHz")
    print(f"  Ratio de supresión necesario: ~{f_mean/F_MAX_WINDOW:.0e}×")
    print(f"  Candidatos: screening geométrico, interferencia CDT, renormalización")
    
    # Guardar CSV
    out = 'GRU_F_SPINE_results.csv'
    with open(out, 'w') as f:
        f.write('file,N,T,N_spine,f_spine,n_edges,n_temporal,n_spatial,n_periodic\n')
        for r in results:
            f.write(f"{r['file']},{r['N']},{r['T']},{r['N_spine']},"
                    f"{r['f_spine']:.8e},{r['n_edges_total']},"
                    f"{r['n_temporal']},{r['n_spatial']},{r['n_periodic']}\n")
    print(f"\nDatos: {out}")


if __name__ == '__main__':
    main(sys.argv[1:])
