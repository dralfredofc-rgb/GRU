# GRU v2.6 — HALLAZGOS PENDIENTES PARA v2.7+

**Documento generado:** 5 de julio de 2026
**Propósito:** Lista completa de bugs, placeholders y mejoras encontradas durante la auditoría de cierre v2.6, para no perder el hilo si se cierra la sesión de chat.

---

## 🔴 CRÍTICO — Corregir en v2.7

### 1. Scripts con amplitudes placeholder/obsoletas

| Script | Problema | Acción requerida | Impacto |
|--------|----------|------------------|---------|
| `GRU_P_ANISOTROPY_v2_5.py` | Usa A=4.235e-8 (hardcodeado, memoria #57, obsoleto) | Reemplazar por A_FINAL=1.46e-16 o derivar de P_PROP_FULL | Mapa direccional de anisotropía |
| `GRU_P_ANISOTROPY__2__FIXED.py` | Placeholder A=1.0e-17, comentario dice "sustituir" | Reemplazar por A_FINAL=1.46e-16 | k_SME, retardos direccionales |
| `GRU_P_WAVEFORM_FIXED.py` | Placeholder A=1.35e-17, comentario dice "sustituir" | Reemplazar por A_FINAL=1.46e-16 | Forma de onda GRU en banda LISA |

**Nota:** Estos scripts NO afectan el HTML v2.6 (no generaron valores publicados), pero sí podrían usarse en el futuro para claims cuantitativos.

### 2. Script eliminado por afirmación falsa

| Script | Razón de eliminación |
|--------|----------------------|
| `GRU_P_DISTINGUISHABILITY_FIXED.py` (v2.3) | Imprime "GRU (min/cen/max) — Detectable LISA", afirmación invalidada por v2.6 |

**Acción:** Verificar que no quede en ningún ZIP/GitHub. Ya eliminado por Hermes.

---

## 🟡 MEDIO — Mejorar en v2.7

### 3. Dependencia de frecuencia del screening M1

**Pregunta abierta:** ¿Podría el screening M1 ser banda-dependiente, permitiendo A_eff > A_FINAL en banda LISA?

**Estado:** Consultado con Kimi y Perplexity (4 jul 2026). Respuesta: NO existe en la literatura (Vainshtein, camaleón, K-mouflage, symmetron) mecanismo para hacer screening dependiente de frecuencia. Estos mecanismos dependen de condiciones locales (densidad, curvatura), no de la frecuencia de la señal observada.

**Acción:** Dejar como pregunta abierta explícita en el paper. No investigar más salvo que surja derivación desde estructura causal de GRU.

### 4. D=3 vs D=4 en Regge Hessian

**Problema:** Los resultados de §A.42 asumen D=4, κ=1/(8π), pero las simulaciones CDT son 2+1D (D=3), κ=1/(6π). Corrección ~33% en λ(Λ).

**Impacto:** El cálculo de LISA falsifiability NO depende del origen microscópico de M1, solo de su valor medido λ_corr=2.14. Pero la derivación formal del Hessiano Regge → MDR sí requiere D=4.

**Acción:** v2.7+ o colaboración con Clemente/Loll para datos 4D.

### 5. CDT 4D real

**Estado:** Sin respuesta de Clemente (joek93, INFN Pisa) ni Loll (Radboud). Deadline autoimpuesto: 9 julio 2026.

**Acción si no hay respuesta:** Publicar v2.6 como está, con nota explícita de que 4D es abierto. Considerar PRD/CQG directo.

---

## 🟢 BAJO — Documentación/infraestructura

### 6. Scripts exploratorios no validados

| Script | Estado | Nota |
|--------|--------|------|
| `GRU_QUANTUM_FILTER.py` v2.0 | OOM con T_obs=1 año | Solo funciona demo T_obs=1 día. Requiere optimización de memoria o procesamiento por chunks. |
| Calculadora A.37 v4.9 | No peer-reviewed | Herramienta interactiva, no incluir en paper. |
| Explorador Dimensional 3D | No peer-reviewed | Herramienta interactiva, no incluir en paper. |

### 7. Documentos PRD/CQG secundarios con menciones LISA sin corregir

| Documento | Línea | Problema |
|-----------|-------|----------|
| `GRU_PRDCQG_Seccion1_Introduction_DRAFT.txt` | ~68-69 | Mención LISA sin contexto de N₅=3.5×10¹⁸ |
| `GRU_CoverLetter_CQG_DRAFT.txt` | ~37 | Mención LISA sin contexto de indetectabilidad cuantitativa |
| `GRU_Discusion_Limitaciones_CONSOLIDADA.txt` | — | Falta añadir limitación LISA/N₅≈3.5×10¹⁸ explícita |

**Acción:** Corregir antes de enviar a PRD/CQG.

### 8. GitHub README y Zenodo descripción

**Estado:** Siguen con "N=1 evento SMBHB a 5σ" (incorrecto).

**Acción:** Actualizar al publicar v2.6 en Zenodo y GitHub.

---

## ✅ VERIFICADO — No requiere acción

| Item | Estado | Evidencia |
|------|--------|-----------|
| Fisher v3 con A_FINAL | ✅ Correcto | `sigma_n = 2.1121e6`, `N5 = 3.506e18` |
| SO(3) ensemble corregido | ✅ Correcto | `SO3_emergent: true` |
| HTML v2.6 valores post-screening | ✅ Correcto | Grep de cierre limpio |
| F.5 documenta corrección | ✅ Correcto | 204 → 3.5e18 documentado |
| d_s, f_spine, PPN, CMB | ✅ Sin cambios | Independientes del bug Fisher |

---

## Contacto
Alfredo Flores Cornejo — dr.alfredo.fc@gmail.com

**Nota para futuras sesiones:** Este documento fue generado por Kimi K2.6 el 5 de julio de 2026, a partir del reporte de auditoría de Hermes (mismo día). Si se retoma en sesión posterior, verificar que los items 🔴 no hayan sido corregidos ya por otro colaborador.
