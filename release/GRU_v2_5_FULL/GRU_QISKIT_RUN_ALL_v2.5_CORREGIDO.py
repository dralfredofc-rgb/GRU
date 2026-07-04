#!/usr/bin/env python3
"""
GRU_QISKIT_RUN_ALL_v2.5_CORREGIDO.py

Pipeline maestro que ejecuta los 6 scripts corregidos y genera
reporte consolidado para Zenodo.

Uso: python3 GRU_QISKIT_RUN_ALL_v2.5_CORREGIDO.py
"""

import subprocess
import json
import time
from datetime import datetime

SCRIPTS = [
    "GRU_QISKIT_GEOMETRIC_CHANNEL_v2.5_CORREGIDO.py",
    "GRU_QISKIT_WALK_BENCHMARK_v2.5_CORREGIDO.py",
    "GRU_QISKIT_TEMPORAL_WALK_v2.5_CORREGIDO.py",
    "GRU_QISKIT_RADIAL_PROJECTION_v2.5_CORREGIDO.py",
    "GRU_QISKIT_GRAPH_STATE_CODE_v2.5_CORREGIDO.py",
    "GRU_QISKIT_HOLOGRAPHIC_QEC_v2.5_CORREGIDO.py"
]

OUTPUTS = [
    "GRU_QISKIT_channel_results_CORREGIDO.json",
    "GRU_QISKIT_walk_benchmark_CORREGIDO.json",
    "GRU_QISKIT_temporal_walk_CORREGIDO.json",
    "GRU_QISKIT_radial_projection_CORREGIDO.json",
    "GRU_QISKIT_graph_state_results_CORREGIDO.json",
    "GRU_QISKIT_holographic_qec_CORREGIDO.json"
]

def main():
    print("=" * 70)
    print("GRU QISKIT v2.5 — PIPELINE DE VALIDACION CUANTICA")
    print("Scripts corregidos con datos GRU reales")
    print("=" * 70)

    results = {}
    start_time = time.time()

    for script, output in zip(SCRIPTS, OUTPUTS):
        print(f"\n{'='*70}")
        print(f"Ejecutando: {script}")
        print(f"{'='*70}")

        try:
            result = subprocess.run(
                ["python3", script],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos maximo por script
            )

            print(result.stdout)
            if result.stderr:
                print(f"[STDERR] {result.stderr[:500]}")

            # Cargar JSON de salida
            try:
                with open(output, "r") as f:
                    data = json.load(f)
                results[script] = {"status": "PASS", "data": data}
            except Exception as e:
                results[script] = {"status": "OUTPUT_ERROR", "error": str(e)}

        except subprocess.TimeoutExpired:
            results[script] = {"status": "TIMEOUT"}
        except Exception as e:
            results[script] = {"status": "ERROR", "error": str(e)}

    elapsed = time.time() - start_time

    # Reporte consolidado
    report = {
        "gru_version": "2.5_CORREGIDO",
        "doi": "10.5281/zenodo.20939080",
        "timestamp": datetime.now().isoformat(),
        "execution_time_seconds": elapsed,
        "scripts_executed": len(SCRIPTS),
        "results_summary": {s: r["status"] for s, r in results.items()},
        "detailed_results": results
    }

    with open("GRU_QISKIT_v2.5_CORREGIDO_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("RESUMEN EJECUCION")
    print("=" * 70)
    for script, result in results.items():
        status = result["status"]
        symbol = "✅" if status == "PASS" else "❌"
        print(f"{symbol} {script}: {status}")

    print(f"\nTiempo total: {elapsed:.1f}s")
    print("Reporte: GRU_QISKIT_v2.5_CORREGIDO_report.json")

if __name__ == "__main__":
    main()
