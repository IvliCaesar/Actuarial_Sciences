"""Descarga y verificación de todos los datos del curso.

    python -m actuarial.descargar            # todo
    python -m actuarial.descargar --cas      # sólo CASdatasets (Dinamarca, Francia)
    python -m actuarial.descargar --mx       # sólo fuentes mexicanas (CENAPRED, CNSF, INPC)
    python -m actuarial.descargar --local    # no descarga: verifica/usa archivos puestos a mano

Ver DATOS.md para la descarga manual desde el navegador.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

from . import data, mexico


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cas(local: bool) -> bool:
    ok = True
    print("CASdatasets (commit fijo, SHA-256 verificado):")
    for name in data.SHA256:
        path = data.DATA_DIR / f"{name}.rda"
        try:
            if local:
                if not path.exists():
                    raise FileNotFoundError(f"falta {path}")
                if _sha(path) != data.SHA256[name]:
                    raise RuntimeError(f"SHA-256 distinto en {path}")
            else:
                data.fetch(name)
            print(f"  OK  {path}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  ERROR  {name}: {e}")
    return ok


def mx(local: bool) -> bool:
    print("Fuentes mexicanas (CENAPRED, CNSF, INPC):")
    reference = json.loads((mexico.OUT / "manifest.json").read_text(encoding="utf-8"))
    try:
        new = mexico.build(force=not local)
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: {e}\n  Descargue los archivos a mano (DATOS.md, sección 3) y use --local.")
        return False
    for name, info in new.items():
        same = reference.get(name, {}).get("sha256") == info["sha256"]
        tag = "idéntico al del repositorio" if same else "DISTINTO al del repositorio (la fuente se actualizó)"
        print(f"  OK  {name}: {info['bytes']:,} bytes, {tag}")
    print(f"  extractos regenerados en {mexico.OUT}")
    return True


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cas", action="store_true")
    ap.add_argument("--mx", action="store_true")
    ap.add_argument("--local", action="store_true")
    a = ap.parse_args()
    both = not (a.cas or a.mx)
    ok = True
    if a.cas or both:
        ok &= cas(a.local)
    if a.mx or both:
        ok &= mx(a.local)
    print("\nListo." if ok else "\nHubo errores; revise DATOS.md.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
