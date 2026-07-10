"""
GRU Challenges Pipeline — v2.5.6+corrected
Ejecuta los 4 scripts .py reales y genera JSON consolidado.
"""

import subprocess
import sys
import json
from datetime import datetime

SCRIPTS = [
    ("Ward Identity", "ward_identity_GRU.py"),
    ("PPN Parameters", "ppn_GRU.py"),
    ("SO(3) Symmetry", "so3_symmetry_GRU.py"),
    ("Ringdown Echoes", "ringdown_echoes_GRU.py"),
]

def main():
    results = {}
    all_pass = True

    print("=" * 70)
    print("GRU v2.5.6+ CHALLENGES PIPELINE")
    print("=" * 70)

    for name, script in SCRIPTS:
        print(f"\n[{name}]")
        print("-" * 40)
        try:
            r = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True, timeout=60
            )
            results[name] = {
                "returncode": r.returncode,
                "status": "PASS" if r.returncode == 0 else "FAIL",
                "stdout": r.stdout[:3000],
                "stderr": r.stderr[:1000] if r.stderr else "",
            }
            if r.returncode != 0:
                all_pass = False
            print(r.stdout[:2000])
        except Exception as e:
            results[name] = {"status": "ERROR", "message": str(e)}
            all_pass = False
            print(f"ERROR: {e}")

    # Guardar JSON consolidado
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "GRU_version": "v2.5.6+corrected",
        "all_pass": all_pass,
        "results": results,
    }

    with open("GRU_challenges_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETADO")
    print("=" * 70)
    print(f"JSON guardado: GRU_challenges_results.json")
    print(f"Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")

    return all_pass

if __name__ == "__main__":
    status = main()
    sys.exit(0 if status else 1)
