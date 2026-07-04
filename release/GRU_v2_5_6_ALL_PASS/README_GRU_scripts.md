# GRU v2.5.6+ — Scripts de Validación
## Instrucciones de instalación y ejecución en Ubuntu

### Requisitos
- Ubuntu 20.04+ (testeado en 26.04)
- Python 3.10+
- numpy, scipy, matplotlib

### Paso 1: Descargar los archivos
Descarga estos dos archivos en tu Ubuntu:
1. `GRU_scripts_v2.5.6.zip` — contiene los 5 scripts Python
2. `install_gru_scripts.sh` — script de instalación automática

### Paso 2: Ejecutar instalación
```bash
bash install_gru_scripts.sh
```

Esto:
- Verifica/instala dependencias (numpy, scipy, matplotlib)
- Crea directorio `~/GRU_v2_5_6_scripts/`
- Descomprime los scripts
- Crea `run_all.sh` para ejecución completa

### Paso 3: Ejecutar scripts

**Todos a la vez:**
```bash
bash run_all.sh
```

**Individualmente:**
```bash
python3 ward_identity_GRU.py      # Unitariedad/Ward
python3 ppn_GRU.py                 # Parámetros PPN
python3 so3_symmetry_GRU.py        # Simetría SO(3)
python3 ringdown_echoes_GRU.py     # Ecos en ringdown
python3 GRU_challenges_pipeline.py # Pipeline maestro
```

### Qué genera cada script

| Script | Output | Tiempo estimado |
|--------|--------|-----------------|
| `ward_identity_GRU.py` | `GRU_ward_identity.png` + consola | ~5s |
| `ppn_GRU.py` | Tabla en consola | ~2s |
| `so3_symmetry_GRU.py` | Tabla + promedio ensemble | ~3s |
| `ringdown_echoes_GRU.py` | `GRU_ringdown_potential.png` + `GRU_echo_template.png` + tabla | ~10s |
| `GRU_challenges_pipeline.py` | Reporte consolidado | ~20s (ejecuta los 4) |

### Estructura de los scripts

```
~/GRU_v2_5_6_scripts/
├── ward_identity_GRU.py          # §2 — Identidades de Ward
├── ppn_GRU.py                     # §3 — Límite PPN
├── so3_symmetry_GRU.py            # §4 — Simetría SO(3)
├── ringdown_echoes_GRU.py         # §5 — Ecos en ringdown
├── GRU_challenges_pipeline.py     # §6 — Pipeline maestro
├── run_all.sh                     # Ejecuta todo automáticamente
└── GRU_scripts_v2.5.6.zip         # Backup del ZIP original
```

### Parámetros GRU usados

```python
f_spine  = 7.752e-3   # Hz
Delta_t  = 128.35  # s  # histórico: 0.129 s
d_s_low  = 1.03       # Zona I (espina)
d_s_high = 5.02       # Zona III (bulk QG)
A_GRU    = 4.235e-8   # Coeficiente MDR
n_GRU    = 0.0564     # Exponente transición
```

### Predicción más fuerte

**Ecos en ringdown con Δt = 128.35 s fijo** para agujeros negros de masa estelar (30-100 M⊙), detectable con datos LIGO existentes.

---
*GRU Framework v2.5.6+ | 2026-06-30*
