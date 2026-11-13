"""Ejecuta todo el estudio: notas, reaseguro con datos reales y supervivencia.

    python -m actuarial            # escribe results/ y figures/
"""
from __future__ import annotations

import contextlib
import io
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from . import charts, data, reinsurance, survival

ROOT = Path(__file__).resolve().parents[1]


def notes_example() -> str:
    """Corre ``notas/optimizacion_xl.py`` y devuelve su salida textual."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(ROOT / "notas" / "optimizacion_xl.py"), run_name="__main__")
    return buf.getvalue()


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run_all(save: bool = True, n_years: int = 50_000) -> dict:
    charts.enable()
    out: dict = {"notas": notes_example()}

    danish = data.load_danish()
    re = reinsurance.run(danish, n_years=n_years)
    out["reaseguro"] = re

    fire = data.load_fire_claims()
    sv = survival.run(fire)
    out["supervivencia"] = sv

    figs = {
        "01_capa_xl": charts.layer_anatomy(380, 1500, 3500),
        "02_frecuencia_danesa": charts.annual_counts(re["counts"]),
        "03_cola_severidad": charts.exceedance(re["emp"], re["fits"]),
        "04_exceso_medio": charts.mean_excess(re["mean_excess"], re["summary"]["gpd_u"]),
        "05_costo_capa": charts.cost_heatmap(re["grid"], re["best"]),
        "06_perdida_anual": charts.annual_distribution(re["dist"]),
        "06b_frontera": charts.frontier(re["frontier"], re["best"]),
        "07_supervivencia_ajuste": charts.survival_fit(sv["curves"]),
        "08_supervivencia_perfiles": charts.survival_profiles(sv["profiles"]),
        "09_calibracion": charts.hazard_calibration(sv["calibration"]),
        "10_brier_ipcw": charts.brier_bars(sv["brier"]),
    }
    out["figuras"] = figs

    if save:
        for name, ch in figs.items():
            charts.save(ch, ROOT / "figures" / f"{name}.png")
        res = ROOT / "results"
        res.mkdir(exist_ok=True)
        (res / "notas_salida.txt").write_text(out["notas"], encoding="utf-8")
        summary = {"reaseguro": re["summary"], "supervivencia": sv["summary"],
                   "reserva_descontada": sv["reserve"],
                   "metricas_supervivencia": sv["metrics"].to_dict(orient="records"),
                   "brier_ipcw": sv["brier"].to_dict(orient="records"),
                   "validacion_prima_capa": re["layer_check"].to_dict(orient="records")}
        (res / "resumen.json").write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    return out


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 140)
    out = run_all()
    print(out["notas"])
    print(json.dumps(_jsonable(out["reaseguro"]["summary"]), indent=2, ensure_ascii=False))
    print(out["reaseguro"]["layer_check"].to_string(index=False))
    print(json.dumps(out["supervivencia"]["summary"], indent=2, ensure_ascii=False))
    print(out["supervivencia"]["metrics"].to_string(index=False))
    print(json.dumps(_jsonable(out["supervivencia"]["reserve"]), indent=2, ensure_ascii=False))
    print("Figuras en figures/, resumen en results/resumen.json")
