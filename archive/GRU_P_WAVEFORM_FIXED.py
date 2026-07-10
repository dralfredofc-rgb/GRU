# DEPRECATED (v2.6, 5 jul 2026): placeholder A=1.35e-17 no validado
# No usar para valores publicados. Ver audit/GRU_AUDITORIA_CIERRE_v2_6_REPORTE.txt

#!/usr/bin/env python3
"""
GRU-LISA P_WAVEFORM: Plantilla de forma de onda GRU para LISA
===========================================================
Autor: Alfredo Flores Cornejo — GRU v2.3
Compatible: LALSuite / LISA Data Challenge (LDC)

DESCRIPCION:
------------
La forma de onda GRU es la de Relatividad General multiplicada por
un factor de fase que acumula el dephasing de la MDR:

    h_GRU(f) = h_GR(f) * exp(i * DeltaPsi_GRU(f))

    DeltaPsi(f) = -2*pi*f * Delta_t(f,z)

ESTRUCTURA:
-----------
- h_plus, h_cross: polarizaciones GR estandar
- DeltaPsi_GRU(f): fase acumulada por dispersion radial
- Parametro libre: A_GRU (dentro de ventana [5.27e-20, 4.36e-14])

NOTA: Plantilla conceptual. Implementacion completa en LALSuite
requiere integracion en C con LALSimulation.
"""
import math, os, os, numpy as np

try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz
OUTPUT_DIR = os.environ.get('GRU_OUTPUT_DIR', '.')


H0 = 70e3 / 3.086e22
EPl = 1.956e9 * 1.602e-10
hP = 6.626e-34
G = 6.674e-11
Msun = 1.989e30
c = 2.998e8
Om, OL = 0.30, 0.70

N_GRU = 0.0564
DS_SPINE = 1.0282
A_GRU = 1.35e-17

M1_Msun = 1e6
M2_Msun = 1e6
Z_GW = 1.0

def K_cosmo(z, n, N=800):
    zz = np.linspace(0, z, N)
    Eh = np.sqrt(Om*(1+zz)**3 + OL)
    return trapz((1+zz)**n / Eh, zz)

def D_L_Gpc(z):
    zz = np.linspace(0, z, 800)
    Eh = np.sqrt(Om*(1+zz)**3 + OL)
    D_C = (3e5/70) * trapz(1/Eh, zz) / 1000
    return D_C * (1 + z)

def chirp_mass(M1, M2):
    return (M1 * M2)**0.6 / (M1 + M2)**0.4 * Msun

def h_GR_amplitude(f, M1, M2, z):
    Mc = chirp_mass(M1, M2)
    D_L = D_L_Gpc(z) * 3.086e25
    h0 = (G * Mc / c**3)**(5/6) * (math.pi * f)**(-7/6) * (5/24)**0.5 / (math.pi**2 * D_L)
    return h0

def Delta_t_GRU(f, z, A=A_GRU, n=N_GRU):
    pre = (1 + n) / (2 * H0)
    Eg = hP * f
    K = K_cosmo(z, n)
    return pre * A * (Eg / EPl)**n * K

def DeltaPsi_GRU(f, z, A=A_GRU, n=N_GRU):
    return -2 * math.pi * f * Delta_t_GRU(f, z, A, n)

def h_GRU(f, M1, M2, z, A=A_GRU, n=N_GRU):
    h0 = h_GR_amplitude(f, M1, M2, z)
    dpsi = DeltaPsi_GRU(f, z, A, n)
    return h0 * np.exp(1j * dpsi)

if __name__ == "__main__":
    print("="*65)
    print("P_WAVEFORM: Plantilla de onda GRU para LISA")
    print("Compatible: LALSuite / LISA Data Challenge")
    print("="*65)

    print("\n1. Parametros de la fuente:")
    print("   M1 = M2 = %.0e M_sun" % M1_Msun)
    print("   z = %.1f" % Z_GW)
    print("   D_L = %.2f Gpc" % D_L_Gpc(Z_GW))
    print("   M_chirp = %.2e M_sun" % (chirp_mass(M1_Msun, M2_Msun)/Msun))

    print("\n2. Parametros GRU:")
    print("   n_GRU = %.4f" % N_GRU)
    print("   A_GRU = %.2e" % A_GRU)

    print("\n3. Tabla de dephasing (z=%.1f):" % Z_GW)
    print("   f(mHz)   Delta_t(ms)   DeltaPsi(rad)   h0(strain)")
    print("   " + "-"*60)
    for fmhz in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        f = fmhz * 1e-3
        dt = Delta_t_GRU(f, Z_GW) * 1e3
        dpsi = DeltaPsi_GRU(f, Z_GW)
        h0 = h_GR_amplitude(f, M1_Msun, M2_Msun, Z_GW)
        print("   %6.1f   %12.4e   %14.4e   %12.4e" % (fmhz, dt, dpsi, h0))

    print("\n4. Comparacion con GR pura:")
    f_test = 3e-3
    dpsi_gru = DeltaPsi_GRU(f_test, Z_GW)
    print("   Dephasing GRU @ 3 mHz: %.4e rad" % dpsi_gru)
    print("   Umbral LISA (SNR=1000): ~0.1 rad dephasing detectable")
    if abs(dpsi_gru) > 0.1:
        print("   => Detectable por LISA")
    else:
        print("   => Sub-umbral (ajustar A_GRU)")

    print("\n5. Estructura para LALSuite:")
    print("   h_GRU(f) = h_GR(f) * exp(i * DeltaPsi_GRU(f))")
    print("   Parametro libre: A_GRU en [%.2e, %.2e]" % (5.27e-20, 4.36e-14))
    print("   Implementacion C: ver LALSimInspiral.c")

    print("\n6. PENDIENTE:")
    print("   Integrar en LALSimulation como LALSimInspiralGRU")
    print("   Validar contra LISA Data Challenge (LDC)")
    print("   Sustituir A_GRU placeholder con valor derivado")
