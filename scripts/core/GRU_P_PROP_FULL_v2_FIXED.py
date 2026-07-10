#!/usr/bin/env python3
"""
GRU-LISA P_PROP_FULL v2: Propagador del graviton tensorial spin-2
==================================================================
Autor: Alfredo Flores Cornejo — GRU v2.3
Referencia: Dittrich, Freidel, Speziale (2007), arXiv:0707.4513

CORRECCION v2:
-------------
1. Ruta de salida configurable (OUTPUT_DIR)
2. Lee N_spine y cos_radial desde P_TSCAN_3D_result.json si existe
3. Calcula g_eff con datos reales cuando estan disponibles
"""
import math, numpy as np, itertools, json, sys, os

OUTPUT_DIR = os.environ.get("GRU_OUTPUT_DIR", ".")

KAPPA = 1.0/(8*math.pi)
DS_SPINE = 1.0282
N_GRU = 2*DS_SPINE - 2

N_SPINE_REAL = None
COS_RADIAL_REAL = None
if os.path.exists("P_TSCAN_3D_result.json"):
    try:
        with open("P_TSCAN_3D_result.json") as f:
            tscan = json.load(f)
            N_SPINE_REAL = tscan.get("N_spine_mean")
            COS_RADIAL_REAL = tscan.get("cos_radial_mean")
            print("[INFO] Datos TSCAN cargados: N_spine=%s, cos_radial=%s" % (N_SPINE_REAL, COS_RADIAL_REAL))
    except:
        pass

def volumes_equilateral(ell=1.0):
    V4 = math.sqrt(5)/96 * ell**4
    Vi = math.sqrt(2)/12 * ell**3
    Vij = math.sqrt(3)/4 * ell**2
    return V4, Vi, Vij

def dihedral_equilateral():
    cos_t = 0.25
    return math.acos(cos_t), cos_t, math.sin(math.acos(cos_t))

def C_tensor_DFS(cos_ij, cos_ik, cos_il, cos_jk, cos_jl):
    return (cos_ik*cos_il + cos_jk*cos_jl)*cos_ij + cos_ik*cos_jl + cos_jk*cos_il

def dtheta_dell_equilateral(n_dim=4, ell=1.0):
    theta, cos_t, sin_t = dihedral_equilateral()
    V4, Vi, Vij = volumes_equilateral(ell)
    c = cos_t
    C_val = C_tensor_DFS(c, c, c, c, c)
    return (1.0/n_dim**2) * (ell/sin_t) * (Vi*Vi/V4**2) * C_val

def build_hessian_equilateral(ell=1.0):
    vertices = [0,1,2,3,4]
    edges = list(itertools.combinations(vertices, 2))
    edge_idx = {e: i for i, e in enumerate(edges)}
    N_edges = len(edges)

    dtheta_dl = dtheta_dell_equilateral(n_dim=4, ell=ell)
    dAh_dl = ell/math.sqrt(3)
    hinges_per_edge = 3

    def n_shared_hinges(e1, e2):
        union = set(e1) | set(e2)
        if len(union) == 2: return hinges_per_edge
        elif len(union) == 3: return 1
        else: return 0

    H = np.zeros((N_edges, N_edges))
    for i, e1 in enumerate(edges):
        for j, e2 in enumerate(edges):
            n_sh = n_shared_hinges(e1, e2)
            if i == j:
                H[i,j] = -KAPPA * hinges_per_edge * dtheta_dl * dAh_dl
            elif n_sh > 0:
                H[i,j] = -KAPPA * n_sh * dtheta_dl * dAh_dl
    return H, edges, edge_idx

def g_eff_from_hessian(H, edges, edge_idx, spine_edges, cos_radial=0.5):
    spine_idx = [edge_idx[e] for e in spine_edges]
    N_spine = len(spine_edges)
    H_spine = H[np.ix_(spine_idx, spine_idx)]
    R = np.full((N_spine, N_spine), cos_radial)
    np.fill_diagonal(R, 1.0)
    g_eff_val = KAPPA * np.sum(H_spine * R) / N_spine
    return g_eff_val, H_spine, R, N_spine

def A_GRU_from_g_eff(g_eff_val, ds=DS_SPINE):
    F_ds = math.gamma(ds/2) / 2.0
    return g_eff_val**2 * F_ds, F_ds

def lambda_corr_needed(A_GRU_val, N_spine_real, A_target):
    ratio = math.log(A_target / A_GRU_val)
    if ratio >= 0 or -2*N_spine_real/ratio <= 0:
        return float('nan')
    return math.sqrt(-2*N_spine_real / ratio)

if __name__ == "__main__":
    print("="*65)
    print("P_PROP_FULL v2: g_eff del graviton desde Hessiano Regge 4D")
    print("="*65)

    ell = 1.0
    H, edges, edge_idx = build_hessian_equilateral(ell)
    theta, cos_t, sin_t = dihedral_equilateral()
    V4, Vi, Vij = volumes_equilateral(ell)
    dth = dtheta_dell_equilateral()

    print("\n1. 4-simplex equilatero: d(theta)/d(ell) = %.8f" % dth)

    evals = np.sort(np.linalg.eigvalsh(H))
    print("   Eigenvalores: %.6f ... %.6f" % (evals[0], evals[-1]))

    if N_SPINE_REAL and COS_RADIAL_REAL:
        print("\n2. Usando datos reales TSCAN:")
        print("   N_spine = %s" % N_SPINE_REAL)
        print("   cos_radial = %s" % COS_RADIAL_REAL)
        spine_edges = [(0, i+1) for i in range(min(int(N_SPINE_REAL), len(edges)))]
        g_eff, H_sp, R_sp, N_sp = g_eff_from_hessian(H, edges, edge_idx, spine_edges, COS_RADIAL_REAL)
    else:
        print("\n2. Modo juguete (N_spine=4):")
        spine_edges = [(0,1), (0,2), (0,3), (0,4)]
        g_eff, H_sp, R_sp, N_sp = g_eff_from_hessian(H, edges, edge_idx, spine_edges)

    print("   g_eff = %.6e" % abs(g_eff))
    g_p11 = N_sp/len(edges)/(8*math.pi)
    print("   |g_eff/g_P1.1| = %.4f" % (abs(g_eff)/g_p11))

    A_GRU, F_ds = A_GRU_from_g_eff(g_eff)
    print("\n3. A_GRU = %.6e" % A_GRU)

    A_min_LISA = 5.27e-20
    A_max_LISA = 4.36e-14
    print("\n4. Ventana LISA: [%.2e, %.2e]" % (A_min_LISA, A_max_LISA))

    for N_test in [N_sp, 50, 100, 500, 1000]:
        lam = lambda_corr_needed(A_GRU, N_test, A_max_LISA)
        status = "EN VENTANA" if 1.46 <= lam <= 2.90 else "FUERA"
        print("   N=%4d: lambda=%.4f slices %s" % (N_test, lam, status))

    result = {
        "g_eff": float(abs(g_eff)), "A_GRU": float(A_GRU),
        "F_dS": float(F_ds), "n_GRU": float(N_GRU),
        "N_spine_used": N_sp,
        "cos_radial_used": float(COS_RADIAL_REAL) if COS_RADIAL_REAL else 0.5,
        "status": "real_data" if N_SPINE_REAL else "juguete",
        "nota": "Sustituir con datos reales de P_TSCAN_3D"
    }

    out_path = os.path.join(OUTPUT_DIR, "P_PROP_FULL_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("\n[GUARDADO] %s" % out_path)
