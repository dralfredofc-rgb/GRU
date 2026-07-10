#!/usr/bin/env python3
"""
GRU-LISA TABLE_LISA: Tabla Δt(f,z) para Apéndice C
====================================================
Alfredo Flores Cornejo — GRU v2.x

Genera la función de transferencia Δt(f,z) en ms y ms/Gpc.
Esta es la tabla que los científicos de LISA necesitan para buscar
la firma de GRU en los datos (publicar antes de 2037).

ENTRADA: A_GRU derivado por F_SPINE_post (editar CONFIG abajo)
         Si no tienes el valor real, usa A_MAX=1.35e-17 (límite superior)

SALIDA:  GRU_LISA_table.txt y GRU_LISA_table.csv

USO: python3 GRU_TABLE_LISA.py
     python3 GRU_TABLE_LISA.py 1.35e-17    (pasar A_GRU como argumento)
"""
import sys, math
import numpy as np

try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz

# ── CONSTANTES ──────────────────────────────────────────────────────────────
H0    = 70e3 / 3.086e22
EPl   = 1.956e9 * 1.602e-10
hP    = 6.626e-34
Om, OL = 0.30, 0.70

# ── CONFIG: ajustar con valores derivados reales ─────────────────────────────
DS_SPINE = 1.0282
N_GRU    = 2 * DS_SPINE - 2   # 0.0564 — DERIVADO, no ajustado

# A_GRU: reemplazar con el valor de F_SPINE_post.py cuando lo tengas
# Por ahora: límite superior de la ventana (A_MAX de P1.5)
A_GRU_DEFAULT = 1.35e-17

# Grilla de frecuencias y redshifts
F_LIST_MHZ = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
Z_LIST     = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# Umbral LISA
DT_LISA_MIN_MS = 0.1


def K(z, n, N=800):
    zz = np.linspace(0, z, N)
    Eh = np.sqrt(Om*(1+zz)**3 + OL)
    return trapz((1+zz)**n / Eh, zz)


def delta_t_ms(f_hz, z, A, n=N_GRU):
    Eg = hP * f_hz
    return ((1+n)/(2*H0)) * A * (Eg/EPl)**n * K(z, n) * 1e3


def D_C_Gpc(z):
    zz = np.linspace(0, z, 800)
    Eh = np.sqrt(Om*(1+zz)**3 + OL)
    return (3e5/70) * trapz(1/Eh, zz) / 1000


def main():
    # A_GRU desde argumento o default
    A_GRU = A_GRU_DEFAULT
    if len(sys.argv) > 1:
        try:
            A_GRU = float(sys.argv[1])
        except ValueError:
            print(f"[AVISO] Argumento inválido, usando A_GRU={A_GRU_DEFAULT:.2e}")
    
    print("="*72)
    print("GRU-LISA TABLE_LISA: Función de Transferencia Δt(f,z)")
    print("="*72)
    print(f"n_GRU  = {N_GRU:.4f}  (derivado: 2·d_S(spine)-2)")
    print(f"A_GRU  = {A_GRU:.4e}")
    print(f"Umbral LISA: {DT_LISA_MIN_MS} ms\n")
    
    rows = []
    for z in Z_LIST:
        dc = D_C_Gpc(z)
        for f_mHz in F_LIST_MHZ:
            dt = delta_t_ms(f_mHz*1e-3, z, A_GRU)
            detectable = dt >= DT_LISA_MIN_MS
            rows.append({
                'z':          z,
                'D_C_Gpc':    dc,
                'f_mHz':      f_mHz,
                'delta_t_ms': dt,
                'per_Gpc':    dt/dc if dc > 0 else 0.0,
                'detectable': detectable,
            })
    
    # Tabla por z
    for z in Z_LIST:
        dc = D_C_Gpc(z)
        print(f"z={z:.1f}  D_C={dc:.2f} Gpc")
        print(f"  {'f(mHz)':>8}  {'Δt(ms)':>14}  {'Δt/Gpc':>14}  LISA?")
        print("  " + "─"*50)
        for r in rows:
            if r['z'] == z:
                ok = "✅" if r['detectable'] else "❌"
                print(f"  {r['f_mHz']:8.1f}  {r['delta_t_ms']:14.4e}"
                      f"  {r['per_Gpc']:14.4e}  {ok}")
        print()
    
    # Tabla comparativa con otras teorías
    print("─"*70)
    print("TABLA COMPARATIVA (f=3mHz, z=1):")
    print(f"  {'Teoría':<28} {'n':>6}  {'A':>14}  {'Δt(ms)':>14}  Estado")
    print("  " + "─"*72)
    dc1 = D_C_Gpc(1.0)
    casos = [
        ("RG pura",              0,     0,         0.0),
        ("MDR estándar (n=1)",   1,     1.0,       None),
        ("GRU A_max (n=0.056)",  N_GRU, A_GRU,    None),
        ("GRU A_min (n=0.056)",  N_GRU, 5.27e-20,  None),
        ("Gravitón masivo",       None,  None,      89.5),
    ]
    for nombre, n_t, A_t, dt_v in casos:
        if dt_v is None and A_t is not None:
            dt_v = delta_t_ms(3e-3, 1.0, A_t, n=n_t if n_t else 0)
        A_s  = f"{A_t:.3e}" if isinstance(A_t, float) else "─"
        n_s  = f"{n_t:.3f}" if isinstance(n_t, float) else str(n_t) if n_t else "─"
        dt_s = f"{dt_v:.4e}" if isinstance(dt_v, float) else "─"
        ok   = "✅" if isinstance(dt_v, float) and dt_v >= 0.1 else "❌"
        print(f"  {nombre:<28} {n_s:>6}  {A_s:>14}  {dt_s:>14}  {ok}")
    
    # Guardar archivos
    txt_path = 'GRU_LISA_table.txt'
    with open(txt_path, 'w') as f:
        f.write(f"# GRU-LISA Función de Transferencia Δt(f,z)\n")
        f.write(f"# n_GRU={N_GRU:.4f}, A_GRU={A_GRU:.4e}\n")
        f.write(f"# Columnas: z, D_C_Gpc, f_mHz, delta_t_ms, ms_per_Gpc, detectable\n")
        for r in rows:
            f.write(f"{r['z']:.3f} {r['D_C_Gpc']:.6f} {r['f_mHz']:.3f} "
                    f"{r['delta_t_ms']:.8e} {r['per_Gpc']:.8e} "
                    f"{'1' if r['detectable'] else '0'}\n")
    
    csv_path = 'GRU_LISA_table.csv'
    with open(csv_path, 'w') as f:
        f.write('z,D_C_Gpc,f_mHz,delta_t_ms,ms_per_Gpc,detectable\n')
        for r in rows:
            f.write(f"{r['z']:.6f},{r['D_C_Gpc']:.6f},{r['f_mHz']:.6f},"
                    f"{r['delta_t_ms']:.8e},{r['per_Gpc']:.8e},"
                    f"{'1' if r['detectable'] else '0'}\n")
    
    print(f"\nGuardado: {txt_path} y {csv_path}")


if __name__ == '__main__':
    main()
