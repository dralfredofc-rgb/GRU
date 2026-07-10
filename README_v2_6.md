# GRU v2.6 — README

Fecha: 10 de julio de 2026
DOI v2.6: 10.5281/zenodo.21288365
DOI Concept: 10.5281/zenodo.20352929
**Versión anterior:** v2.5 (DOI: 10.5281/zenodo.21144855)

---

## Correcciones principales respecto a v2.5

Esta versión corrige DOS hallazgos independientes identificados en auditoría
post-publicación de v2.5: (1) el bug de amplitud pre/post-screening en el
cálculo de Fisher para LISA, y (2) una comparación mal formulada contra la
cota Fermi-LAT de GRB 090510 (issue "C6").

v2.6 corrige un **bug crítico en el cálculo de Fisher Information** para
detectabilidad LISA (amplitud pre-screening usada por error en lugar de
la post-screening):

| Parámetro | v2.5 (incorrecto) | v2.6 (corregido) |
|-----------|-------------------|-------------------|
| Amplitud usada | A_PRE = 1.9148×10⁻⁸ (pre-screening) | A_FINAL = 1.46×10⁻¹⁶ (post-screening) |
| σ_n (1 evento) | 0.0027 | 2.1121×10⁶ |
| N₅ (eventos 5σ) | 204 | 3.506×10¹⁸ |
| Separación vs LQG (n=2) | 720σ | 9.2×10⁻⁷σ |
| Separación vs SME d=5 (n=1) | 349σ | 4.47×10⁻⁷σ |

**Conclusión:** Con el screening M1 aplicado universalmente (posición
conservadora dado el estado actual del marco), GRU es **cualitativamente
distinguible** (n_GRU<1 + ventana de amplitud LISA + anisotropía
cuadripolar única) pero **cuantitativamente indetectable** por LISA en su
misión primaria. Ver §7.4.8 y Apéndice F.5 del paper.

También se corrigió un bug de muestreo (no isotrópico) en el ensemble del
test SO(3), sin cambio en la conclusión física publicada (breaking
~10⁻⁸² a escala LIGO, vía fórmula analítica independiente).

---

## Contenido del ZIP (este listado refleja el contenido real, 1:1)

### Raíz
- `GRU_v2_6.html` — Paper corregido y verificado (final)
- `README_v2_6.md` — Este archivo
- `HALLAZGOS_PENDIENTES_v2_6.md` — Hallazgos de la auditoría de cierre

### `audit/` — Evidencia de la corrección
- `GRU_AUDITORIA_CIERRE_v2_6_REPORTE.txt` — Reporte completo (4–5 jul 2026)
- `GRU_AUDITORIA_SESION_4_5_julio_2026.txt` — Bitácora de la sesión de auditoría
- `GRU_AUDITORIA_AMPLITUDES_CONSUMIDORES.py` — Escáner de amplitudes
- `GRU_AUDITORIA_AMPLITUDES_result.json` — Resultado del escaneo
- `GRU_P_FISHER_v3_result.json` — Resultado Fisher con A_FINAL (evidencia)
- `GRU_SO3_result_CORREGIDO.json` — Resultado SO(3) con muestreo corregido
- `GRU_v26_grep_cierre.sh` — Script de verificación de cierre del HTML
- `GRU_v26_grep_cierre_output_20260710.txt` — Output de la verificación final de cierre, guardado como evidencia con fecha de publicación
- `GRU_C6_FERMI_LAT_DIFERENCIAL.py` — Recomputo diferencial de la cota Fermi-LAT
- `GRU_C6_FERMI_LAT_DIFERENCIAL_result.json` — Resultado numérico (9 combinaciones método/banda)

### `scripts/core/` — Pipeline principal y QC
- `GRU_Dplus_operator_v2.py`, `GRU_Dplus_run_CDT_real.py`
- `GRU_F_SPINE_measure.py`
- `GRU_P1_root_invariance_test_FIXED.py`
- `GRU_P1_5_fermi_lat_nGRU_FIXED.py`
- `GRU_P_PROP_FULL_v2_FIXED.py`, `GRU_P_PROP_FULL_v2_5_fix.py`
- `GRU_QC_CHANNEL_1QUBIT.py`, `GRU_QC_CHANNEL_2QUBIT.py`,
  `GRU_QC_CHOI_ANALYSIS.py`, `GRU_QC_QUANTUM_WALK_SPINE.py`
- `GRU_QISKIT_*_v2.5_CORREGIDO.py` (8 scripts QC/Qiskit) y
  `GRU_QISKIT_GEOMETRIC_CHANNEL_v2.5_GRU_FIEL.py`
- `GRU_QUANTUM_FILTER_v2.5.4.py` (última versión; anteriores en `archive/`)
- `GRU_challenges_pipeline_CORREGIDO.py`, `GRU_pipeline_ALL_CORRECTED.py`
- `ppn_GRU_CORREGIDO.py`, `ringdown_echoes_GRU_CORREGIDO.py`,
  `ward_identity_GRU_CORREGIDO.py`

### `scripts/mdr_lisa/` — Fenomenología LISA/MDR
- `GRU_P_FISHER_v3_OFICIAL_A_FINAL.py` — **Fisher oficial con A_FINAL correcto**
- `GRU_FISHER_VERIFICACION_INDEPENDIENTE.py` — Verificación independiente del bug
- `GRU_P_DISTINGUISHABILITY_v2_1_CORREGIDO.py`
- `GRU_P1_MDR_CONSISTENCY.py`
- `GRU_QC_MDR_DEPHASING.py`
- `GRU_CMB_LISA_final.py`
- `GRU_GWTC_anisotropy_test_CORREGIDO.py`
- `GRU_LISA_SGWB_SPECTRUM.py`, `GRU_LISA_TIME_DELAY_TEST.py`
- `GRU_TABLE_LISA.py`

### `scripts/ppn_so3/` — Gravedad clásica / simetrías
- `GRU_SO3_symmetry_v2_CORREGIDO_muestreo.py` — **SO(3) con muestreo isotrópico corregido**
- `ward_identity_GRU_v3.py`, `ward_identity_GRU_v3.1.py`

### `future/` — Roadmap
- `HALLAZGOS_PENDIENTES_v2_7.md`

### `archive/` — Obsoletos (cada uno con cabecera `# DEPRECATED` interna)
- `GRU_P_FISHER_v2_FIXED.py` — Versión con el bug de amplitud; superseded por v3. Conservado como referencia histórica del bug.
- `so3_symmetry_GRU_CORREGIDO.py` — Versión con bug de muestreo no isotrópico; su `SO3_emergent:false` era artefacto. Superseded por `GRU_SO3_symmetry_v2_CORREGIDO_muestreo.py`.
- `GRU_P_ANISOTROPY_v2_5.py` — A=4.235e-8 pre-screening obsoleto. La razón ε_A=4.1816% publicada es matemáticamente independiente de la amplitud (verificado).
- `GRU_P_WAVEFORM_FIXED.py` — Placeholder A=1.35e-17, no usado en valores publicados.
- `GRU_QUANTUM_FILTER.py`, `GRU_QUANTUM_FILTER_v2.5.1.py`, `GRU_QUANTUM_FILTER_v2.5.2.py` — Versiones anteriores a v2.5.4.

**Nota explícita:** `GRU_P_DISTINGUISHABILITY_FIXED.py` (v2.3) fue
**eliminado por completo**, no archivado. Imprimía la afirmación
"Detectable LISA", invalidada por esta corrección. No está en este ZIP
ni en el repositorio.

---

## Estado de validación

| Componente | Estado | Nota |
|------------|--------|------|
| d_s(spine) ≈ 1 | ✅ Validado | 60 geom 2D, multiseed 3D, toy 4D |
| CMB quadrupole | ✅ Compatible | D_ℓ = 6.34 μK² (+0.67σ Planck) |
| LISA amplitude | ✅ En ventana | A_FINAL = 1.46×10⁻¹⁶ ∈ [5.27×10⁻²⁰, 4.36×10⁻¹⁴] |
| LISA detectability | ⚠️ Cualitativa | N₅ ≈ 3.5×10¹⁸ (indetectable cuantitativamente) |
| PPN γ | ✅ Compatible | δγ ~2×10⁴ veces debajo del límite Cassini |
| SO(3) ensemble | ✅ Restaurado | Breaking ~10⁻⁸² (analítico); ensemble isotrópico tras corrección |
| Root Invariance | ✅ Validado | d_s = 0.9602 ± 0.0413 (20 spines, §A.46) |
| CDT 4D real | ❌ Abierto | Sin respuesta de Clemente/Loll a la fecha |

---

## Falsifiabilidad (v2.6)

1. **Replicación independiente** de d_s(spine) ≈ 1 en 2D/3D CDT (protocolo en este ZIP)
2. **LiteBIRD (2028+)** — si mide D_ℓ consistente con ΛCDM
3. **LISA (2037+)** — si no detecta la anisotropía cuadripolar ε_A ≈ 4.18%
   (la separación estadística cuantitativa vs. teorías rivales no es
   alcanzable con el presupuesto de eventos de la misión primaria;
   ver §7.4.8 y F.5)

---

## Pendientes conocidos (no bloqueantes; ver HALLAZGOS_PENDIENTES_v2_6.md)

- 3 documentos secundarios PRD/CQG aún sin la corrección LISA — revisar antes de someter.
- GitHub README y descripción de Zenodo con cifra de v2.5 — actualizar al publicar.
- Screening M1 banda-dependiente: pregunta abierta sin mecanismo en la literatura.
- CDT 4D real: pendiente de Clemente/Loll.

---

## Contacto
Alfredo Flores Cornejo — dr.alfredo.fc@gmail.com
Zapopan, Jalisco, México
