#!/bin/bash
# GRU_v26_grep_cierre.sh
# ==================================================================
# Verificacion de cierre para GRU_v2_6_DRAFT.html.
# Ejecutar DESPUES de aplicar las correcciones de sigma_n y de la
# Sec. 7.4.8. Pegar el output COMPLETO y LITERAL en el reporte de
# sesion -- no resumir ni afirmar "quedo limpio" sin el output.
#
# Uso: bash GRU_v26_grep_cierre.sh /ruta/a/GRU_v2_6_DRAFT.html
# ==================================================================

FILE="${1:-GRU_v2_6_DRAFT.html}"

if [ ! -f "$FILE" ]; then
    echo "ERROR: no se encuentra el archivo $FILE"
    exit 1
fi

echo "===================================================================="
echo "VERIFICACION DE CIERRE: $FILE"
echo "===================================================================="

echo ""
echo "--- 1) Menciones residuales de 'N=1' o 'N = 1' en contexto LISA ---"
grep -noE "N ?= ?1[^0-9]" "$FILE" || echo "(ninguna coincidencia -- OK)"

echo ""
echo "--- 2) Todas las apariciones de sigma_n / N5 (revisar consistencia) ---"
grep -n "σ<sub>n</sub>\|sigma_n\|N5\|N<sub>5</sub>\|N<sub>events</sub>" "$FILE"

echo ""
echo "--- 3) Verificar que NO quede sigma_n=0.0027 sin actualizar ---"
grep -n "0\.0027" "$FILE" || echo "(ninguna coincidencia -- OK)"

echo ""
echo "--- 4) Verificar presencia y contenido de la seccion de correccion LISA ---"
grep -n "F\.4\|F\.5\|Correction to LISA" "$FILE"

echo ""
echo "--- 5) Verificar separaciones de discriminabilidad (720/349 sigma viejos) ---"
grep -n "720" "$FILE" || echo "(ninguna coincidencia -- si aparecio antes, confirmar que ya no esta)"
grep -n "349" "$FILE" || echo "(ninguna coincidencia -- si aparecio antes, confirmar que ya no esta)"

echo ""
echo "===================================================================="
echo "FIN DE VERIFICACION. Pegar este output completo en el reporte."
echo "===================================================================="
