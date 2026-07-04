# GRU QISKIT v2.5 — Scripts Corregidos

## Descripción
Paquete de 6 scripts Qiskit corregidos que conectan el pipeline GRU real
con computación cuántica. Todos los parámetros se derivan de datos reales
del servidor (GRU_mesh_perfect.pkl, GRU_P1_1_g_spine_Regge.py).

## Correcciones respecto a v2.5 anterior

| Aspecto | Versión anterior (incorrecta) | Versión corregida |
|---------|------------------------------|-------------------|
| Grafo | `nx.cycle_graph(5)` sintético | Espina real 200 nodos desde PKL |
| dₛ(spine) | Desconocido/arbitrario | **1.0282** (oficial GRU) |
| dₛ(full) | Desconocido | **1.6715** (oficial GRU) |
| ε_A | 0.0412 arbitrario | **0.0169** derivado |dₛ-1|/dₛ(full) |
| f_spine | No usado | **0.0323** (200/6200) |
| Moneda temporal | Coeficientes 0.5, 0.7 arbitrarios | Derivada de shell real |
| IQFT | Hadamard falso | **IQFT real** con rotaciones controladas |
| Heat kernel | Fórmula ad-hoc | **Protocolo GRU A.21** (fit P(σ)~Aσ^(-α)) |
| Graph state | Grafo sintético cadena+bulk | **Subgrafo espina real** |
| Holographic QEC | 4 qubits bulk → 1 qubit boundary arbitrario | **Conectividad real** bulk-boundary |

## Archivos generados

| Script | Descripción | Salida |
|--------|-------------|--------|
| `GRU_QISKIT_GEOMETRIC_CHANNEL_v2.5_CORREGIDO.py` | Canal CPTP desde geometría | `GRU_QISKIT_channel_results_CORREGIDO.json` |
| `GRU_QISKIT_WALK_BENCHMARK_v2.5_CORREGIDO.py` | Heat kernel clásico vs quantum walk | `GRU_QISKIT_walk_benchmark_CORREGIDO.json` |
| `GRU_QISKIT_TEMPORAL_WALK_v2.5_CORREGIDO.py` | Moneda optimizada para foliación | `GRU_QISKIT_temporal_walk_CORREGIDO.json` |
| `GRU_QISKIT_RADIAL_PROJECTION_v2.5_CORREGIDO.py` | Operador R̂ con distancias BFS | `GRU_QISKIT_radial_projection_CORREGIDO.json` |
| `GRU_QISKIT_GRAPH_STATE_CODE_v2.5_CORREGIDO.py` | Graph state sobre espina real | `GRU_QISKIT_graph_state_results_CORREGIDO.json` |
| `GRU_QISKIT_HOLOGRAPHIC_QEC_v2.5_CORREGIDO.py` | Encoding bulk→boundary real | `GRU_QISKIT_holographic_qec_CORREGIDO.json` |
| `GRU_QISKIT_RUN_ALL_v2.5_CORREGIDO.py` | Pipeline maestro | `GRU_QISKIT_v2.5_CORREGIDO_report.json` |

## Requisitos
- Python 3.10+
- Qiskit 2.4.2+
- Qiskit Aer
- NetworkX, NumPy, SciPy
- Archivo `/root/GRU_mesh_perfect.pkl` (datos reales)

## Instalación
```bash
python3 -m venv gru_qc_env
source gru_qc_env/bin/activate
pip install qiskit qiskit-aer networkx numpy scipy
```

## Uso
```bash
# Ejecutar todos los scripts
python3 GRU_QISKIT_RUN_ALL_v2.5_CORREGIDO.py

# O ejecutar individualmente
python3 GRU_QISKIT_GEOMETRIC_CHANNEL_v2.5_CORREGIDO.py
```

## Parámetros derivados de datos reales

```python
DS_SPINE = 1.0282          # dₛ espina (GRU_P1_1_g_spine_Regge.py)
DS_FULL = 1.6715           # dₛ grafo completo (GRU_P1_1_g_spine_Regge.py)
SIGMA_DS = 0.025           # incertidumbre
F_SPINE = 200 / 6200       # 0.0323 fracción nodos espina
N_BULK_PER_SHELL = 30      # nodos bulk por shell
EPSILON_A = 0.0169         # |1.0282 - 1| / 1.6715
COMPRESSION_RATIO = 1.626  # DS_FULL / DS_SPINE
```

## DOI
10.5281/zenodo.20939080 (v2.4)
