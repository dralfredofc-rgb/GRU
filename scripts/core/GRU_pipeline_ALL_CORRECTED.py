"""
GRU Challenges Pipeline — v2.5.6+ALL_CORRECTED
Ejecuta los 4 scripts _CORREGIDO y genera JSON consolidado.
"""

import subprocess
import sys
import json
from datetime import datetime, timezone

SCRIPTS = [
    ("Ward Identity", "ward_identity_GRU_CORREGIDO.py", "GRU_Ward_result.json"),
    ("PPN Parameters", "ppn_GRU_CORREGIDO.py", "GRU_PPN_result.json"),
    ("SO(3) Symmetry", "so3_symmetry_GRU_CORREGIDO.py", "GRU_SO3_result.json"),
    ("Ringdown Echoes", "ringdown_echoes_GRU_CORREGIDO.py", None),
]

def main():
    results = {}
    all_pass = True

    print("=" * 70)
    print("GRU v2.5.6+ ALL CORRECTED CHALLENGES PIPELINE")
    print("=" * 70)

    for name, script, json_file in SCRIPTS:
        print(f"\n[{name}]")
        print("-" * 40)
        try:
            r = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True, timeout=60
            )

            # Leer JSON si existe
            json_data = None
            if json_file:
                try:
                    with open(json_file, "r") as f:
                        json_data = json.load(f)
                except:
                    pass

            results[name] = {
                "returncode": r.returncode,
                "status": "PASS" if r.returncode == 0 else "FAIL",
                "stdout": r.stdout[:2000],
                "json_data": json_data,
            }
            if r.returncode != 0:
                all_pass = False
            print(r.stdout[:1500])
        except Exception as e:
            results[name] = {"status": "ERROR", "message": str(e)}
            all_pass = False
            print(f"ERROR: {e}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "GRU_version": "v2.5.6+ALL_CORRECTED",
        "all_pass": all_pass,
        "results": results,
    }

    with open("GRU_challenges_results_ALL.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETADO")
    print("=" * 70)
    print(f"JSON: GRU_challenges_results_ALL.json")
    print(f"Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")

    for name, data in results.items():
        status = data.get("status", "UNKNOWN")
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {name}: {status}")

    return all_pass

if __name__ == "__main__":
    status = main()
    sys.exit(0 if status else 1)
