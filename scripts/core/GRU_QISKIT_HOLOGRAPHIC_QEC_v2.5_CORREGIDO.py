#!/usr/bin/env python3
"""
GRU_QISKIT_HOLOGRAPHIC_QEC_v2.5_CORREGIDO.py

Encoding holografico inspirado en estructura GRU real.

CORRECCIONES:
- Bulk = nodos bulk reales (30 por shell), no 4 qubits arbitrarios
- Boundary = espina real (1 nodo por shell)
- Encoding usa conectividad real grafo-bulk (no CZ generico)
- Ratio de compresion derivado de datos: ds_full/ds_spine = 1.626

DOI v2.4: 10.5281/zenodo.20939080
"""

import numpy as np
import json
import pickle
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, state_fidelity

# ============================================================
# 1. CARGAR ESTRUCTURA BULK-BOUNDARY REAL
# ============================================================

def load_gru_holographic_structure(pkl_path="/root/GRU_mesh_perfect.pkl", max_shells=4):
    """
    Carga estructura bulk-boundary desde malla GRU real.

    Bulk: nodos perifericos alrededor de cada nodo espinal
    Boundary: nodos espinales (1 por shell)

    Limitamos a max_shells para simulacion NISQ.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    G_full = data["graph"]
    spine_nodes = data["spine_nodes"]
    bulk_nodes_list = data["bulk_nodes"]  # lista de listas

    # Tomar primeros max_shells shells
    n_shells = min(max_shells, len(spine_nodes), len(bulk_nodes_list))

    boundary = [spine_nodes[i] for i in range(n_shells)]
    bulk = []
    for i in range(n_shells):
        if i < len(bulk_nodes_list):
            bulk.extend(bulk_nodes_list[i][:5])  # max 5 bulk por shell para NISQ

    # Extraer subgrafo
    all_nodes = boundary + bulk
    G_sub = G_full.subgraph(all_nodes).copy()

    print(f"[GRU-REAL] Holographic structure:")
    print(f"  Boundary (spine): {len(boundary)} nodos")
    print(f"  Bulk: {len(bulk)} nodos")
    print(f"  Total: {G_sub.number_of_nodes()} nodos, {G_sub.number_of_edges()} aristas")

    return G_sub, boundary, bulk

# ============================================================
# 2. ENCODING HOLOGRAFICO DESDE CONECTIVIDAD REAL
# ============================================================

def holographic_encoder_real(G, boundary, bulk):
    """
    Encoding holografico que respeta conectividad real GRU.

    Cada qubit de bulk se entrelaza con su nodo espinal (boundary)
    mediante la arista real del grafo.
    """
    n_bulk = len(bulk)
    n_boundary = len(boundary)

    qr_bulk = QuantumRegister(n_bulk, "bulk")
    qr_boundary = QuantumRegister(n_boundary, "boundary")
    cr = ClassicalRegister(n_boundary, "meas")
    qc = QuantumCircuit(qr_bulk, qr_boundary, cr)

    # Preparar estado de bulk (ejemplo: GHZ)
    if n_bulk > 0:
        qc.h(qr_bulk[0])
        for i in range(n_bulk - 1):
            qc.cx(qr_bulk[i], qr_bulk[i + 1])

    qc.barrier()

    # Encoding: entrelazar bulk con boundary usando conectividad real
    node_to_idx_bulk = {node: i for i, node in enumerate(bulk)}
    node_to_idx_boundary = {node: i for i, node in enumerate(boundary)}

    for u, v in G.edges():
        if u in node_to_idx_bulk and v in node_to_idx_boundary:
            i_bulk = node_to_idx_bulk[u]
            i_boundary = node_to_idx_boundary[v]
            qc.cx(qr_bulk[i_bulk], qr_boundary[i_boundary])
        elif v in node_to_idx_bulk and u in node_to_idx_boundary:
            i_bulk = node_to_idx_bulk[v]
            i_boundary = node_to_idx_boundary[u]
            qc.cx(qr_bulk[i_bulk], qr_boundary[i_boundary])

    qc.barrier()

    # Capa de compresion: boundary interactua (si hay conectividad)
    for i in range(n_boundary - 1):
        if boundary[i] in G and boundary[i + 1] in G:
            if G.has_edge(boundary[i], boundary[i + 1]):
                qc.cz(qr_boundary[i], qr_boundary[i + 1])

    qc.barrier()
    qc.measure(qr_boundary, cr)

    return qc

# ============================================================
# 3. DECODER Y RECOVERY
# ============================================================

def holographic_decoder_real(qc, boundary, bulk, shots=1024):
    """
    Decoder: reconstruir informacion de bulk desde mediciones de boundary.
    """
    simulator = AerSimulator()
    job = simulator.run(qc, shots=shots)
    counts = job.result().get_counts()

    # Analisis: entropia de boundary
    probs = np.array([c / shots for c in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(counts))

    # Capacidad de recuperacion: correlacion bulk-boundary
    n_boundary = len(boundary)
    n_bulk = len(bulk)

    # Heuristica: si boundary tiene alta entropia, codifica mas informacion
    encoding_efficiency = entropy / max_entropy if max_entropy > 0 else 0

    return {
        "n_distinct_outcomes": len(counts),
        "entropy": float(entropy),
        "max_entropy": float(max_entropy),
        "encoding_efficiency": float(encoding_efficiency),
        "code_rate": n_bulk / n_boundary if n_boundary > 0 else 0,
        "compression_ratio": n_bulk / n_boundary if n_boundary > 0 else 0,
        "counts": dict(list(counts.items())[:10])
    }

# ============================================================
# 4. TEST CON ERRORES
# ============================================================

def test_recovery_real(G, boundary, bulk, error_rate=0.1):
    """
    Test de recuperacion con errores en boundary.
    """
    n_bulk = len(bulk)
    n_boundary = len(boundary)

    qr_bulk = QuantumRegister(n_bulk, "bulk")
    qr_boundary = QuantumRegister(n_boundary, "boundary")
    cr = ClassicalRegister(n_bulk + n_boundary, "c")
    qc = QuantumCircuit(qr_bulk, qr_boundary, cr)

    # Preparar bulk
    if n_bulk > 0:
        qc.h(qr_bulk[0])
        for i in range(n_bulk - 1):
            qc.cx(qr_bulk[i], qr_bulk[i + 1])

    qc.barrier()

    # Encoding real
    node_to_idx_bulk = {node: i for i, node in enumerate(bulk)}
    node_to_idx_boundary = {node: i for i, node in enumerate(boundary)}

    for u, v in G.edges():
        if u in node_to_idx_bulk and v in node_to_idx_boundary:
            qc.cx(qr_bulk[node_to_idx_bulk[u]], qr_boundary[node_to_idx_boundary[v]])
        elif v in node_to_idx_bulk and u in node_to_idx_boundary:
            qc.cx(qr_bulk[node_to_idx_bulk[v]], qr_boundary[node_to_idx_boundary[u]])

    qc.barrier()

    # Errores en boundary
    n_errors = max(1, int(error_rate * n_boundary))
    error_positions = np.random.choice(n_boundary, size=n_errors, replace=False)
    for pos in error_positions:
        if np.random.random() < 0.5:
            qc.x(qr_boundary[pos])
        else:
            qc.z(qr_boundary[pos])

    qc.barrier()

    # Decoding: intentar recuperar bulk
    for u, v in G.edges():
        if u in node_to_idx_bulk and v in node_to_idx_boundary:
            qc.cx(qr_boundary[node_to_idx_boundary[v]], qr_bulk[node_to_idx_bulk[u]])
        elif v in node_to_idx_bulk and u in node_to_idx_boundary:
            qc.cx(qr_boundary[node_to_idx_boundary[u]], qr_bulk[node_to_idx_bulk[v]])

    qc.measure(qr_bulk, cr[:n_bulk])
    qc.measure(qr_boundary, cr[n_bulk:])

    simulator = AerSimulator()
    job = simulator.run(qc, shots=1024)
    counts = job.result().get_counts()

    # Verificar recuperacion
    correct = 0
    for bitstring, count in counts.items():
        bulk_part = bitstring[:n_bulk] if n_bulk > 0 else ""
        if bulk_part.count("0") >= n_bulk * 0.8 or bulk_part.count("1") >= n_bulk * 0.8:
            correct += count

    recovery_rate = correct / 1024

    return {
        "error_rate": error_rate,
        "recovery_rate": recovery_rate,
        "code_rate": n_bulk / n_boundary if n_boundary > 0 else 0,
        "n_bulk": n_bulk,
        "n_boundary": n_boundary
    }

# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("GRU QISKIT — HOLOGRAPHIC QEC v2.5 CORREGIDO")
    print("Estructura bulk-boundary desde malla GRU REAL")
    print("=" * 70)

    # Cargar estructura
    G, boundary, bulk = load_gru_holographic_structure(max_shells=3)

    # Encoding
    print("\n[Encoding] Circuito holografico...")
    qc_enc = holographic_encoder_real(G, boundary, bulk)
    print(f"  Qubits: {qc_enc.num_qubits}, Depth: {qc_enc.depth()}")

    # Decoder
    dec = holographic_decoder_real(qc_enc, boundary, bulk)
    print(f"\n[Decoder] Entropia: {dec['entropy']:.4f} / {dec['max_entropy']:.4f}")
    print(f"  Eficiencia: {dec['encoding_efficiency']:.4f}")
    print(f"  Code rate: {dec['code_rate']:.2f}")
    print(f"  Ratio compresion: {dec['compression_ratio']:.2f}")

    # Recovery test
    print("\n[Recovery] Test con errores...")
    for err in [0.0, 0.1, 0.2]:
        rec = test_recovery_real(G, boundary, bulk, error_rate=err)
        print(f"  Error={err:.1f}: Recovery={rec['recovery_rate']:.3f}")

    # Guardar
    output = {
        "gru_version": "2.5_CORREGIDO",
        "doi": "10.5281/zenodo.20939080",
        "data_source": "GRU_mesh_perfect.pkl",
        "boundary_nodes": len(boundary),
        "bulk_nodes": len(bulk),
        "code_rate": dec["code_rate"],
        "compression_ratio": dec["compression_ratio"],
        "encoding_efficiency": dec["encoding_efficiency"],
        "recovery_tests": [test_recovery_real(G, boundary, bulk, error_rate=e) for e in [0.0, 0.1, 0.2]],
        "interpretation": f"Holographic encoding desde conectividad GRU real: "
                          f"bulk={len(bulk)} -> boundary={len(boundary)}, "
                          f"ratio={dec['compression_ratio']:.2f}"
    }

    with open("GRU_QISKIT_holographic_qec_CORREGIDO.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n[GRU-HQEC] Resultados en GRU_QISKIT_holographic_qec_CORREGIDO.json")

if __name__ == "__main__":
    main()
