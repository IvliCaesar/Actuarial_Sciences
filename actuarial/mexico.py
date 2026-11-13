"""Casos mexicanos con datos oficiales.

Fuentes (descarga directa, sin credenciales):

* CENAPRED, *Base histórica de impacto socioeconómico de desastres 2000-2024*
  (Atlas Nacional de Riesgos): daño total por evento y estado, millones de pesos corrientes.
* CNSF, Sistemas Estadísticos de Riesgos Hidrometeorológicos y de Terremoto
  (portal de Datos Abiertos de la CNSF): siniestros asegurados del mercado
  2015-2025, con monto recuperado de reaseguro desde 2021.
* OCDE vía FRED, serie ``MEXCPIALLMINMEI`` (índice de precios al consumidor de México)
  para expresar los daños en pesos de 2024.

``build()`` descarga las fuentes, guarda su SHA-256 en ``datos_mx/manifest.json``
y escribe extractos derivados pequeños en ``datos_mx/``; ``run()`` sólo lee esos
extractos, de modo que el análisis es reproducible sin red.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import reinsurance as rx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datos_mx"
RAW = ROOT / "data" / "raw_mx"
CNSF = "https://www.cnsf.gob.mx/Transparencia/DGDI/"
SOURCES = {
    "cenapred.xlsx": "http://www.atlasnacionalderiesgos.gob.mx/descargas/Basehistorica_2000_a_2024.xlsx",
    "cnsf_hidro_siniestros.csv": CNSF + urllib.parse.quote("57 Informacion estadistica de Riesgos Hidrometeorologicos_ Siniestros.csv"),
    "cnsf_terremoto_siniestros.csv": CNSF + urllib.parse.quote("60 Informacion estadistica de Terremoto y Erupcion Volcanica_ Siniestros.csv"),
    "cpi_mex.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MEXCPIALLMINMEI",
}
BASE_YEAR = 2024


def _plain(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Construcción de extractos
# ---------------------------------------------------------------------------
def _download() -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in SOURCES.items():
        path = RAW / name
        urllib.request.urlretrieve(url, path)
        manifest[name] = {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "bytes": path.stat().st_size, "descargado": date.today().isoformat()}
    return manifest


def _deflator() -> pd.Series:
    cpi = pd.read_csv(RAW / "cpi_mex.csv")
    cpi["year"] = pd.to_datetime(cpi.iloc[:, 0]).dt.year
    annual = cpi.groupby("year")[cpi.columns[1]].mean()
    return (annual[BASE_YEAR] / annual).rename("factor_a_2024")


_NAME = re.compile(r"(?:hurac[a-z]*|tormenta tropical|depresi[a-z]* tropical|cicl[a-z]* tropical)\s+"
                   r"[\"'“”«»]?([A-Z][a-z]+)", re.IGNORECASE)
_SKIP = {"de", "del", "en", "que", "la", "el", "los", "categoria", "con", "y", "se", "a", "provoco", "ocasiono"}


def _event_name(texts: pd.Series) -> str:
    for t in texts.dropna():
        for m in _NAME.finditer(_plain(t)):
            w = m.group(1)
            if w.lower() not in _SKIP:
                return w.capitalize()
    return ""


def _cenapred(defl: pd.Series) -> pd.DataFrame:
    d = pd.read_excel(RAW / "cenapred.xlsx")
    d.columns = ["inicio", "fin", "anio", "clase", "tipo", "estado", "municipios", "descripcion", "defunciones",
                 "poblacion", "viviendas", "escuelas", "hospitales", "comercios", "cultivo_ha", "dano_mdp", "fuente"]
    d["tipo"] = d["tipo"].map(_plain)
    d["dano_mdp"] = pd.to_numeric(d["dano_mdp"], errors="coerce")
    d["inicio"] = pd.to_datetime(d["inicio"], errors="coerce")
    d["anio"] = pd.to_numeric(d["anio"], errors="coerce")
    d = d[d["tipo"].isin(["Ciclon tropical", "Sismo"]) & (d["dano_mdp"] > 0) & d["inicio"].notna()]
    # Evento = mismo tipo y misma fecha de inicio (un huracán afecta varios estados).
    ev = d.groupby(["tipo", "inicio"]).agg(
        anio=("anio", "first"), dano_mdp=("dano_mdp", "sum"), estados=("estado", lambda s: ", ".join(sorted(set(map(_plain, s))))),
        n_estados=("estado", "size"), defunciones=("defunciones", lambda s: pd.to_numeric(s, errors="coerce").sum()),
        nombre=("descripcion", _event_name)).reset_index()
    ev["anio"] = ev["anio"].astype(int)
    ev["dano_mdp_2024"] = ev["dano_mdp"] * ev["anio"].map(defl)
    return ev.sort_values("inicio").reset_index(drop=True)


def _cnsf() -> tuple[pd.DataFrame, pd.DataFrame]:
    h = pd.read_csv(RAW / "cnsf_hidro_siniestros.csv", encoding="utf-8-sig", low_memory=False)
    t = pd.read_csv(RAW / "cnsf_terremoto_siniestros.csv", encoding="utf-8-sig", low_memory=False)
    h["peligro"] = np.where(h["tipo_evento"].map(_plain).str.startswith("Ciclon"), "Huracán (ciclón tropical)",
                            "Otros hidrometeorológicos")
    t["peligro"] = "Terremoto"
    cols = ["numero_siniestros", "monto_siniestro", "monto_pagado", "monto_reaseguro"]
    both = pd.concat([h[["anio", "entidad", "peligro"] + cols], t[["anio", "entidad", "peligro"] + cols]])
    both["entidad"] = both["entidad"].map(_plain).replace({"Distrito Federal": "Ciudad de Mexico"})
    for c in cols:
        both[c] = pd.to_numeric(both[c], errors="coerce").fillna(0)
    by_year = both.groupby(["anio", "peligro"], as_index=False)[cols].sum()
    by_state = both[both["peligro"] == "Huracán (ciclón tropical)"].groupby("entidad", as_index=False)[cols].sum()
    return by_year, by_state


def build() -> dict:
    """Descarga las fuentes oficiales y regenera ``datos_mx/``."""
    manifest = _download()
    defl = _deflator()
    OUT.mkdir(exist_ok=True)
    ev = _cenapred(defl)
    ev.to_csv(OUT / "cenapred_eventos_ciclon_sismo.csv", index=False)
    by_year, by_state = _cnsf()
    by_year.to_csv(OUT / "cnsf_siniestros_por_anio.csv", index=False)
    by_state.to_csv(OUT / "cnsf_huracan_por_entidad.csv", index=False)
    defl.reset_index().rename(columns={"index": "anio", "year": "anio"}).to_csv(OUT / "deflactor_2024.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------
def load() -> dict:
    if not (OUT / "cenapred_eventos_ciclon_sismo.csv").exists():
        build()
    return {
        "eventos": pd.read_csv(OUT / "cenapred_eventos_ciclon_sismo.csv", parse_dates=["inicio"]),
        "cnsf_anio": pd.read_csv(OUT / "cnsf_siniestros_por_anio.csv"),
        "cnsf_entidad": pd.read_csv(OUT / "cnsf_huracan_por_entidad.csv"),
    }


def run(theta: float = .25, gamma: float = .01, alpha: float = .99, budget_ratio: float = .75,
        u_quantile: float = .70, cap_multiple: float = 3.0, n_years: int = 100_000, seed: int = 2026) -> dict:
    D = load()
    ev = D["eventos"]
    cy = ev[ev["tipo"] == "Ciclon tropical"].copy()
    x = cy["dano_mdp_2024"].to_numpy()
    years = np.arange(2000, 2025)

    counts = cy.groupby("anio").size().reindex(years, fill_value=0).rename("n").reset_index().rename(columns={"index": "Year", "anio": "Year"})
    freq = rx.poisson_trend(counts, center=2012.0)
    lam = float(counts["n"].mean())  # sin tendencia significativa: se usa la media
    disp = float(counts["n"].var(ddof=1) / counts["n"].mean())

    u = float(np.quantile(x, u_quantile))
    cap = cap_multiple * float(x.max())
    sev = rx.fit_severity(x, u=u, seed=seed, cap=cap)
    emp, fits = rx.exceedance_tables(x, sev)
    fits = fits[~fits["modelo"].str.startswith("Pareto")]  # Pareto con x_m = mínimo no es informativa aquí
    fits["modelo"] = fits["modelo"].str.replace("Empírica + GPD", "Empírica + GPD truncada")

    annual = cy.groupby("anio")["dano_mdp_2024"].sum().reindex(years, fill_value=0)
    d_grid = np.unique(np.round(np.geomspace(np.quantile(x, .5), .5 * cap, 30), -2))
    m_grid = np.r_[0, np.unique(np.round(np.geomspace(1000, cap, 30), -2))]
    sim = rx.simulate(sev["spliced"], lam, n_years, d_min=float(d_grid.min()), seed=seed)
    es_gross = rx.expected_shortfall(sim.gross, alpha)
    grid, best = rx.optimize_layer(sim, d_grid, m_grid, theta, gamma, alpha, budget_ratio * es_gross)
    frontier = []
    for r in np.linspace(.5, .98, 25):
        f = grid[grid["ES_net"] <= r * es_gross]
        if len(f):
            b = f.loc[f["J"].idxmin()]
            frontier.append({"ratio": r, "B": r * es_gross, "costo_transferencia": b["J"] - sim.gross.mean(),
                             "d": b["d"], "m": b["m"]})

    cn = D["cnsf_anio"]
    hur = cn[cn["peligro"] == "Huracán (ciclón tropical)"].copy()
    hur["cesion"] = np.where(hur["monto_siniestro"] > 0, hur["monto_reaseguro"] / hur["monto_siniestro"], np.nan)

    econ = cy.groupby("anio")["dano_mdp"].sum()
    ins = hur.set_index("anio")["monto_siniestro"] / 1e6
    yrs = [y for y in range(2015, 2025) if y in ins.index]
    gap = pd.concat([
        pd.DataFrame({"anio": yrs, "serie": "Daño económico (CENAPRED)", "mdp": econ.reindex(yrs, fill_value=0).values}),
        pd.DataFrame({"anio": yrs, "serie": "Siniestro asegurado (CNSF)", "mdp": ins.reindex(yrs).values}),
    ])
    gap = gap[gap["mdp"] > 0]
    ratio = float(ins.reindex(yrs).sum() / econ.reindex(yrs, fill_value=0).sum())

    top = cy.sort_values("dano_mdp_2024", ascending=False).head(10)
    otis = cy.loc[cy["dano_mdp_2024"].idxmax()]
    summary = {
        "eventos_ciclon": int(len(cy)), "periodo": "2000–2024", "lambda_eventos_por_anio": lam,
        "dispersion_var_media": disp, "tendencia_anual_poisson": freq["growth"],
        "umbral_gpd_mdp2024": u, "pml_truncacion_mdp2024": cap, "gpd_xi": sev["xi"], "gpd_xi_ic95": sev["xi_ci"], "excesos": sev["n_exceed"],
        "dano_medio_anual_mdp2024": float(annual.mean()), "evento_maximo": f"{otis['nombre'] or 'sin nombre'} ({otis['inicio']:%Y-%m-%d})",
        "evento_maximo_mdp2024": float(otis["dano_mdp_2024"]),
        "E_S_simulada": float(sim.gross.mean()), f"ES_{alpha}_bruta": es_gross, "B": budget_ratio * es_gross,
        "d_opt": best["d"], "m_opt": best["m"], "prima_opt": best["prima"], f"ES_{alpha}_neta_opt": best["ES_net"],
        "asegurado_entre_economico_2015_2024": ratio,
        "cnsf_huracan_2023_siniestro_mdp": float(hur.loc[hur.anio == 2023, "monto_siniestro"].sum() / 1e6),
        "cnsf_huracan_2023_reaseguro_mdp": float(hur.loc[hur.anio == 2023, "monto_reaseguro"].sum() / 1e6),
    }
    return {"eventos": cy, "annual": annual.rename("dano").reset_index().rename(columns={"anio": "anio"}),
            "counts": counts, "emp": emp, "fits": fits, "grid": grid, "best": best,
            "frontier": pd.DataFrame(frontier), "cnsf_anio": cn, "cnsf_huracan": hur,
            "cnsf_entidad": D["cnsf_entidad"], "top": top, "gap": gap, "summary": summary, "sismos": ev[ev["tipo"] == "Sismo"]}


def sensitivity(caps=(2.0, 3.0, 5.0), u_quantiles=(.6, .7, .8), n_years: int = 40_000) -> pd.DataFrame:
    """Capa óptima ante la pérdida máxima probable (truncación) y el umbral de la GPD."""
    rows = []
    for c in caps:
        for q in u_quantiles:
            s = run(cap_multiple=c, u_quantile=q, n_years=n_years)["summary"]
            rows.append({"PML / máximo observado": c, "cuantil del umbral u": q, "ξ": s["gpd_xi"],
                         "d*": s["d_opt"], "m*": s["m_opt"], "prima": s["prima_opt"]})
    return pd.DataFrame(rows)
