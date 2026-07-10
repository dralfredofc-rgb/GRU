#!/usr/bin/env python3
"""
GRU v2.6 - Test de anisotropia GWTC (H1)
Script CORREGIDO del protocolo original de Grok/Kimi.
El codigo original de Grok tenia una funcion inexistente:
  hp.pixelfunc.get_map_from_points()  <- NO EXISTE en healpy real
Reemplazada por el metodo estandar: hp.ang2pix() + np.bincount()

Requiere: pip install healpy astropy pandas --break-system-packages
"""
import numpy as np
import pandas as pd
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u

# ============================================================
# PASO 1: Cargar datos GWTC reales (reemplazar cuando se ejecute)
# URL verificada funcional:
#   https://gwosc.org/api/v2/catalogs/GWTC/events?include-default-parameters=true&format=csv
# ============================================================
def load_gwtc_data(csv_path='gwtc.csv'):
    df = pd.read_csv(csv_path)
    df = df[df['p_astro'] >= 0.9]  # solo eventos confidentes
    return df

# ============================================================
# PASO 2: Construir mapa HEALPix de eventos (CORREGIDO)
# ============================================================
def build_event_map(ra_deg, dec_deg, nside=32):
    """
    Convierte posiciones RA/Dec a un mapa de conteos HEALPix.
    Reemplaza la funcion inexistente hp.pixelfunc.get_map_from_points.
    """
    npix = hp.nside2npix(nside)
    theta = np.radians(90 - dec_deg)  # colatitud desde declinacion
    phi = np.radians(ra_deg)
    pix_indices = hp.ang2pix(nside, theta, phi)
    map_counts = np.bincount(pix_indices, minlength=npix).astype(float)
    return map_counts

# ============================================================
# PASO 3: Tests de anisotropia (dipolo + cuadripolo)
# ============================================================
def test_dipole(ra_deg, dec_deg):
    coords = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
    vecs = coords.cartesian.xyz.value.T
    dipole = np.mean(vecs, axis=0)
    return np.linalg.norm(dipole)

def test_quadrupole(map_counts):
    cl = hp.sphtfunc.anafast(map_counts)
    return cl  # cl[2] es el termino cuadripolar C_2

def monte_carlo_null(n_events, nside=32, n_trials=1000, seed=42):
    """Genera distribucion nula isotropica para comparar (SIN patron
    de antena todavia -- ese es el siguiente refinamiento pendiente,
    ver Paso 7 del protocolo de Kimi)."""
    rng = np.random.default_rng(seed)
    dipole_null = []
    c2_null = []
    for _ in range(n_trials):
        ra = rng.uniform(0, 360, n_events)
        dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n_events)))
        dipole_null.append(test_dipole(ra, dec))
        map_c = build_event_map(ra, dec, nside)
        cl = test_quadrupole(map_c)
        c2_null.append(cl[2])
    return np.array(dipole_null), np.array(c2_null)

# ============================================================
# MAIN (usar con datos reales cuando se ejecute)
# ============================================================
if __name__ == '__main__':
    NSIDE = 32
    N_MC = 1000

    # --- Placeholder con datos sinteticos para validar el pipeline ---
    print("MODO DEMO (datos sinteticos) -- reemplazar con GWTC real")
    np.random.seed(1)
    n_events = 391  # GWTC-5.0 aproximado
    ra_demo = np.random.uniform(0, 360, n_events)
    dec_demo = np.degrees(np.arcsin(np.random.uniform(-1, 1, n_events)))

    dipole_amp = test_dipole(ra_demo, dec_demo)
    map_counts = build_event_map(ra_demo, dec_demo, NSIDE)
    cl = test_quadrupole(map_counts)

    print(f"Dipolo observado: {dipole_amp:.4f}")
    print(f"C_2 (cuadripolo) observado: {cl[2]:.6f}")

    print(f"\nGenerando {N_MC} simulaciones Monte Carlo nulas...")
    dipole_null, c2_null = monte_carlo_null(n_events, NSIDE, N_MC)

    p_dipole = np.mean(dipole_null >= dipole_amp)
    p_c2 = np.mean(c2_null >= cl[2])

    print(f"\np-value dipolo: {p_dipole:.4f}")
    print(f"p-value cuadripolo: {p_c2:.4f}")
    print(f"\nInterpretacion: p>0.05 => compatible con isotropia")

    # Sensibilidad esperada (verificado analiticamente por Claude, 3 jul)
    A2_detectable_3sigma = 3/np.sqrt(n_events) * 100
    print(f"\nSensibilidad esperada (3sigma, N={n_events}): "
          f"A_2 ~ {A2_detectable_3sigma:.1f}%")
    print("NOTA: este demo usa datos sinteticos aleatorios, no GWTC real.")
    print("PENDIENTE (protocolo completo de Kimi): incorporar patron de")
    print("antena LVK antes de comparar contra datos reales (Paso 7).")
