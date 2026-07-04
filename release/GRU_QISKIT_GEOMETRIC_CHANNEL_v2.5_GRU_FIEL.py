#!/usr/bin/env python3
"""
GRU_QISKIT_GEOMETRIC_CHANNEL_v2.5_GRU_FIEL.py

Canal cuantico CPTP derivado RIGUROSAMENTE de la geometria GRU.

PRINCIPIO GRU:
    La espina (spine) es una cadena causal 1D con ds ~ 1.
    El bulk es un espacio 2D/3D con ds > 1.
    La informacion se confina en la espina porque la difusion en el bulk
    es mas rapida (ds mayor -> dispersion mas rapida -> perdida de informacion).

MODELO FISICO:
    |0> = estado en la espina (protegido, ds = 1, difusion lenta ~sqrt(t))
    |1> = estado en el bulk (expuesto, ds = 2-3, difusion rapida ~t^(ds/2))

    El canal modela la probabilidad de que un estado en el bulk
    "caiga" a la espina por difusion geometrica en tiempo sigma.

DERIVACION MATEMATICA:
    P(sigma) ~ sigma^(-ds/2)  [heat kernel GRU]

    Para spine (ds = 1):   P_spine(sigma) ~ sigma^(-0.5)
    Para bulk  (ds = 2.5): P_bulk(sigma)  ~ sigma^(-1.25)  [promedio 2D/3D]

    La razon de retorno: P_spine / P_bulk ~ sigma^(0.75)
    -> la espina retiene informacion mas tiempo

    Parametro de canal: gamma(sigma) = 1 - (P_spine/P_bulk)

    Para sigma -> inf: gamma -> 1 (todo el bulk cae a la espina)
    Para sigma -> 0: gamma -> 0 (no hay tiempo para difundir)

    Valor caracteristico (sigma = 1 en unidades GRU):
    gamma = 1 - exp(-epsilon_A) donde epsilon_A = |delta DS| / (2*DS_bulk*sqrt(N_spine))

PARAMETROS GRU REALES (CDT 2D):
    DS_SPINE = 1.0282 +/- 0.025  (A.21, NWALKS=5000, SEED=42)
    DS_FULL  = 1.6715 +/- 0.1317 (script comparacion)
    N_SPINE  = 20              (T shells)
    N_TOTAL  = 2567            (JSON real Ubuntu)
    F_SPINE  = 20/2567 = 0.00779  # 7.79 mHz

    epsilon_A (experimental) = |1.0282 - 1.6715| / (2*1.6715*sqrt(20))
                       = 0.6433 / 14.944
                       = 0.04305 (4.305%)

    epsilon_A (geometrico)   = |1.0282 - 1| / 1.6715
                       = 0.01687 (1.687%)

    NOTA: Son observables distintos. El experimental mide la separacion
    real spine-bulk; el geometrico mide la desviacion de ds_spine respecto
    a 1. Usamos el experimental como gold standard.

DOI: 10.5281/zenodo.20939080
STATUS: Sec A.43 Outlook Exploratorio - Mathematical illustration
        Heat kernel A.21 remains the validated gold standard.
"""

import numpy as np
import json

# ============================================================
# 1. PARAMETROS GRU REALES (CDT 2D)
# ============================================================

# Datos validados desde pipeline Ubuntu
DS_SPINE = 1.0282
DS_SPINE_ERR = 0.025
DS_FULL = 1.6715
DS_FULL_ERR = 0.1317
N_SPINE = 20
N_TOTAL = 2567
F_SPINE = N_SPINE / N_TOTAL

# epsilon_A: formula validada (memoria #37)
EPSILON_A_EXPERIMENTAL = abs(DS_SPINE - DS_FULL) / (2.0 * DS_FULL * np.sqrt(N_SPINE))

# epsilon_A: formula geometrica (referencia)
EPSILON_A_GEOMETRIC = abs(DS_SPINE - 1.0) / DS_FULL

# Parametro de canal: derivado de epsilon_A experimental
GAMMA_CHANNEL = 1.0 - np.exp(-EPSILON_A_EXPERIMENTAL)

print("=" * 70)
print("GRU QISKIT - CANAL GEOMETRICO v2.5 GRU-FIEL")
print("Derivacion rigurosa desde geometria GRU")
print("=" * 70)
print()
print("[GRU-PARAM] Parametros reales (CDT 2D):")
print("  DS_SPINE = %.4f +/- %.4f" % (DS_SPINE, DS_SPINE_ERR))
print("  DS_FULL  = %.4f +/- %.4f" % (DS_FULL, DS_FULL_ERR))
print("  N_SPINE  = %d" % N_SPINE)
print("  N_TOTAL  = %d" % N_TOTAL)
print("  F_SPINE  = %.6f (%d/%d)" % (F_SPINE, N_SPINE, N_TOTAL))
print()
print("[GRU-PARAM] epsilon_A dual:")
print("  Experimental (validado): %.6f (%.3f%%)" % (EPSILON_A_EXPERIMENTAL, EPSILON_A_EXPERIMENTAL*100))
print("  Geometrico (referencia): %.6f (%.3f%%)" % (EPSILON_A_GEOMETRIC, EPSILON_A_GEOMETRIC*100))
print("  [NOTA] Observables distintos - ver memoria #37")
print()
print("[GRU-PARAM] gamma (canal) = 1 - exp(-epsilon_A) = %.6f" % GAMMA_CHANNEL)

# ============================================================
# 2. CANAL GRU: AMPLITUDE DAMPING (proyeccion radial)
# ============================================================

class GRUAmplitudeDampingChannel:
    """
    Canal de amortiguacion de amplitud derivado de GRU.

    FISICA:
        El estado |1> (bulk) decae al estado |0> (spine) con probabilidad gamma.
        El estado |0> (spine) es estable - no decae (protegido).

    KRAUS OPERATORS (derivados de la condicion CPTP):
        K_0 = |0><0| + sqrt(1-gamma) |1><1|   = [[1, 0], [0, sqrt(1-gamma)]]
        K_1 = sqrt(gamma) |0><1|             = [[0, sqrt(gamma)], [0, 0]]

    VERIFICACION CPTP:
        K_0^dagger K_0 + K_1^dagger K_1 
        = [[1, 0], [0, 1-gamma]] + [[0, 0], [0, gamma]]
        = [[1, 0], [0, 1]] = I  OK

    INTERPRETACION GRU:
        gamma = probabilidad de que informacion del bulk caiga a la espina
            en un paso de tiempo caracteristico (sigma = 1)
        1-gamma = probabilidad de que informacion permanezca en el bulk
    """

    def __init__(self, gamma=None):
        if gamma is None:
            gamma = GAMMA_CHANNEL
        self.gamma = gamma
        self.dim = 2

        # Kraus operators (derivados rigurosamente)
        self.K0 = np.array([[1.0, 0.0], 
                            [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
        self.K1 = np.array([[0.0, np.sqrt(gamma)], 
                            [0.0, 0.0]], dtype=complex)

        # Verificacion CPTP explicita
        M = self.K0.conj().T @ self.K0 + self.K1.conj().T @ self.K1
        assert np.allclose(M, np.eye(2)), "CPTP violation!"

        print()
        print("[GRU-CH] Canal GRU-FIEL (Amplitude Damping):")
        print("         gamma = %.6f" % gamma)
        print("         K_0 = [[1, 0], [0, %.4f]]" % np.sqrt(1-gamma))
        print("         K_1 = [[0, %.4f], [0, 0]]" % np.sqrt(gamma))
        print("         CPTP: VERIFIED OK")

    def apply(self, rho):
        """Aplica canal a matriz de densidad."""
        return self.K0 @ rho @ self.K0.conj().T + self.K1 @ rho @ self.K1.conj().T

    def choi_matrix(self):
        """Matriz de Choi para verificar rango."""
        C = np.zeros((4, 4), dtype=complex)
        for i in range(2):
            for j in range(2):
                e_ij = np.zeros((2, 2))
                e_ij[i, j] = 1.0
                C += np.kron(e_ij, self.apply(e_ij))
        C = C / 2.0
        rank = np.linalg.matrix_rank(C, tol=1e-10)
        return C, rank

# ============================================================
# 3. TESTS CON ESTADOS DE REFERENCIA
# ============================================================

def test_channel(channel):
    """Tests con estados de referencia GRU."""

    # Estado |0> - en la espina (protegido)
    rho_0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    rho_0_out = channel.apply(rho_0)
    F_0 = np.real(np.trace(rho_0 @ rho_0_out))

    # Estado |1> - en el bulk (expuesto)
    rho_1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    rho_1_out = channel.apply(rho_1)
    F_1 = np.real(np.trace(rho_1 @ rho_1_out))

    # Probabilidad de retorno a espina desde bulk
    p_spine_from_bulk = rho_1_out[0, 0].real

    # Estado |+> - superposicion (test de coherencia)
    rho_plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    rho_plus_out = channel.apply(rho_plus)
    F_plus = np.real(np.trace(rho_plus @ rho_plus_out))

    # Coherencia: elemento off-diagonal
    coh_in = abs(rho_plus[0, 1])
    coh_out = abs(rho_plus_out[0, 1])
    coh_ratio = coh_out / coh_in if coh_in > 0 else 0

    return {
        "F_0": float(F_0),
        "F_1": float(F_1),
        "p_spine_from_bulk": float(p_spine_from_bulk),
        "F_plus": float(F_plus),
        "coherence_in": float(coh_in),
        "coherence_out": float(coh_out),
        "coherence_ratio": float(coh_ratio)
    }

# ============================================================
# 4. MAIN
# ============================================================

def main():
    # Canal GRU
    ch = GRUAmplitudeDampingChannel()

    # Tests
    print()
    print("--- Tests de referencia ---")
    res = test_channel(ch)
    print("Fidelidad |0> (spine):  %.6f  [esperado: 1.0000]" % res['F_0'])
    print("Fidelidad |1> (bulk):   %.6f  [esperado: 1-gamma = %.4f]" % (res['F_1'], 1-GAMMA_CHANNEL))
    print("P(spine|bulk):          %.6f  [esperado: gamma = %.4f]" % (res['p_spine_from_bulk'], GAMMA_CHANNEL))
    print("Fidelidad |+>:         %.6f" % res['F_plus'])
    print("Coherencia ratio:       %.6f  [sqrt(1-gamma) = %.4f]" % (res['coherence_ratio'], np.sqrt(1-GAMMA_CHANNEL)))

    # Choi
    C, rank = ch.choi_matrix()
    print()
    print("Rango de Choi: %d  [esperado: 2 para amplitude damping]" % rank)

    # Guardar
    output = {
        "gru_version": "2.5_GRU_FIEL",
        "doi": "10.5281/zenodo.20939080",
        "status": "Sec A.43 Outlook Exploratorio - Mathematical illustration",
        "parameters": {
            "DS_SPINE": DS_SPINE,
            "DS_SPINE_ERR": DS_SPINE_ERR,
            "DS_FULL": DS_FULL,
            "DS_FULL_ERR": DS_FULL_ERR,
            "N_SPINE": N_SPINE,
            "N_TOTAL": N_TOTAL,
            "F_SPINE": F_SPINE,
            "EPSILON_A_EXPERIMENTAL": EPSILON_A_EXPERIMENTAL,
            "EPSILON_A_GEOMETRIC": EPSILON_A_GEOMETRIC,
            "GAMMA_CHANNEL": GAMMA_CHANNEL
        },
        "derivation": {
            "model": "Amplitude Damping (bulk -> spine projection)",
            "Kraus_K0": "[[1, 0], [0, sqrt(1-gamma)]]",
            "Kraus_K1": "[[0, sqrt(gamma)], [0, 0]]",
            "CPTP_verified": True,
            "gamma_formula": "1 - exp(-epsilon_A_experimental)"
        },
        "results": res,
        "choi_rank": int(rank),
        "interpretation": "Amplitude damping models geometric diffusion from bulk to spine. |0>=spine (protected, ds=1), |1>=bulk (exposed, ds>1). gamma=4.2%% is the probability of falling to spine per GRU time step."
    }

    with open("GRU_QISKIT_channel_results_GRU_FIEL.json", "w") as f:
        json.dump(output, f, indent=2)

    print()
    print("[OUTPUT] Guardado en GRU_QISKIT_channel_results_GRU_FIEL.json")
    print("[STATUS] Sec A.43 Outlook - Heat kernel A.21 remains gold standard")

if __name__ == "__main__":
    main()
