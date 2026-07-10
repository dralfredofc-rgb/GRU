#!/usr/bin/env python3
"""
GRU_AUDITORIA_AMPLITUDES_CONSUMIDORES.py
==================================================================
Escanea un directorio de scripts .py del proyecto GRU y reporta que
valor(es) de amplitud usa cada uno: A_GRU pre-screening (1.9148e-08
o 4.235e-08), A_final post-screening (1.46e-16), u otro valor no
identificado. Sirve para detectar si el bug de amplitud (el mismo
que afecto a GRU_P_FISHER_v2_FIXED.py y a la Sec. 7.4.8 del HTML)
esta presente en otros scripts.

Uso:
    python3 GRU_AUDITORIA_AMPLITUDES_CONSUMIDORES.py /ruta/a/scripts/

Si no se da ruta, escanea el directorio actual.

NOTA: esto es deteccion textual (grep estructurado), no ejecuta los
scripts. Complementa, no sustituye, la lectura manual del codigo.
==================================================================
"""
import sys
import os
import re
import glob
import json

A_PRE_KNOWN = {"1.9148e-08", "1.9147864316914836e-08", "4.235e-08", "4.235e-8"}
A_FINAL_KNOWN = {"1.46e-16", "1.460e-16", "1.4600e-16", "1.460392e-16"}

PATTERN_ASSIGN = re.compile(
    r"(A_GRU|A_FINAL|A_final|A_bruto|A_INT)\s*=\s*([0-9.eE+-]+)"
)
PATTERN_KEY = re.compile(r"['\"](A_GRU|A_final|A_bruto)['\"]\s*:\s*([0-9.eE+-]+)")


def scan_file(path):
    findings = []
    try:
        with open(path, "r", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        return [{"error": str(e)}]

    for m in PATTERN_ASSIGN.finditer(text):
        findings.append({"var": m.group(1), "valor_texto": m.group(2)})
    for m in PATTERN_KEY.finditer(text):
        findings.append({"var": m.group(1), "valor_texto": m.group(2)})
    return findings


def clasificar(valor_texto):
    v = valor_texto.strip()
    if v in A_PRE_KNOWN:
        return "PRE-SCREENING (posible bug si se usa para Fisher/deteccion)"
    if v in A_FINAL_KNOWN:
        return "POST-SCREENING (correcto)"
    try:
        fv = float(v)
        if abs(fv - 1.9148e-08) / 1.9148e-08 < 0.01:
            return "PRE-SCREENING (posible bug)"
        if abs(fv - 1.46e-16) / 1.46e-16 < 0.01:
            return "POST-SCREENING (correcto)"
    except ValueError:
        pass
    return "VALOR NO RECONOCIDO -- revisar manualmente"


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(target, "*.py")))

    report = {}
    print("=" * 78)
    print("AUDITORIA DE AMPLITUDES A_GRU / A_final EN SCRIPTS GRU")
    print("=" * 78)

    for path in files:
        name = os.path.basename(path)
        findings = scan_file(path)
        if not findings:
            continue
        print(f"\n--- {name} ---")
        report[name] = []
        for f in findings:
            if "error" in f:
                print(f"  [ERROR leyendo archivo: {f['error']}]")
                continue
            clase = clasificar(f["valor_texto"])
            print(f"  {f['var']:10s} = {f['valor_texto']:20s}  -> {clase}")
            report[name].append(
                {"variable": f["var"], "valor": f["valor_texto"], "clasificacion": clase}
            )

    with open("GRU_AUDITORIA_AMPLITUDES_result.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("Revisa manualmente cualquier archivo marcado 'PRE-SCREENING' que")
    print("alimente una afirmacion de deteccion/discriminabilidad/Fisher.")
    print("JSON guardado: GRU_AUDITORIA_AMPLITUDES_result.json")
    print("=" * 78)
