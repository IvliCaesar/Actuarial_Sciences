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

from . import charts, data, mexico, problemas, reinsurance, survival

ROOT = Path(__file__).resolve().parents[1]


def _run_script(name: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(ROOT / "notas" / name), run_name="__main__")
    return buf.getvalue()


def notes_example() -> str:
    """Corre ``notas/optimizacion_xl.py`` y devuelve su salida textual."""
    return _run_script("optimizacion_xl.py")


def problem_solutions() -> str:
    """Corre ``notas/soluciones_problemas.py`` (Problemas 1-6)."""
    return _run_script("soluciones_problemas.py")


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

    mx = mexico.run()
    mx["sensibilidad"] = mexico.sensitivity()
    out["mexico"] = mx

    out["soluciones"] = problem_solutions()
    pb = problemas.run()
    out["problemas"] = pb

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
        "11_mx_dano_anual": charts.mx_annual_damage(mx["annual"], mx["top"]),
        "12_mx_cola_ciclones": charts.exceedance(mx["emp"], mx["fits"], unit="millones de pesos de 2024",
                                                 title="Ciclones tropicales en México: cola del daño por evento"),
        "13_mx_costo_capa": charts.cost_heatmap(mx["grid"], mx["best"], unit="mdp 2024",
                                                title="México: costo esperado J(d, m) de una capa XL por ciclón"),
        "14_mx_frontera": charts.frontier(mx["frontier"], mx["best"], unit="mdp 2024",
                                          title="México: frontera costo–riesgo de la capa XL por ciclón"),
        "15_mx_reaseguro_cnsf": charts.mx_insured_vs_ceded(mx["cnsf_huracan"]),
        "16_mx_brecha": charts.mx_protection_gap(mx["gap"]),
        "17_mx_entidades": charts.mx_states(mx["cnsf_entidad"]),
        "18_qq_dinamarca": charts.qq_plot(pb["dinamarca"]["qq"], "QQ log-log: incendios daneses", "mDKK"),
        "19_qq_mexico": charts.qq_plot(pb["mexico"]["qq"], "QQ log-log: ciclones tropicales en México", "mdp 2024"),
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
        (res / "soluciones_salida.txt").write_text(out["soluciones"], encoding="utf-8")
        summary = {"reaseguro": re["summary"], "mexico": mx["summary"],
                   "mexico_sensibilidad": mx["sensibilidad"].to_dict(orient="records"), "supervivencia": sv["summary"],
                   "reserva_descontada": sv["reserve"],
                   "metricas_supervivencia": sv["metrics"].to_dict(orient="records"),
                   "brier_ipcw": sv["brier"].to_dict(orient="records"),
                   "validacion_prima_capa": re["layer_check"].to_dict(orient="records"),
                   "problema7": {k: {"frecuencia": v["frecuencia"], "xi": v["xi"], "xi_entrenamiento": v["xi_train"],
                                     "central": v["central"].to_dict(orient="records"),
                                     "backtest": v["backtest"].to_dict(orient="records"),
                                     "exceso_medio": v["exceso_medio"].to_dict(orient="records")}
                                 for k, v in pb.items()}}
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
