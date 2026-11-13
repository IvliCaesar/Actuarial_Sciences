"""Gráficas Altair con un tema sobrio y consistente.

Paleta categórica validada (orden fijo, nunca ciclado; a lo más tres series por
gráfica), rejilla recesiva, marcas delgadas, tooltips en cada marca.

``enable()`` registra un renderer que emite a la vez la especificación Vega-Lite
(interactiva en JupyterLab/VS Code) y un PNG (visible en GitHub, que no ejecuta
JavaScript en notebooks).
"""
from __future__ import annotations

import base64
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e9e8e4", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
W, H = 560, 300


def _theme():
    return {"config": {
        "background": SURFACE,
        "font": "Inter, Segoe UI, Helvetica, Arial, sans-serif",
        "view": {"stroke": None},
        "title": {"fontSize": 15, "fontWeight": 600, "color": INK, "anchor": "start",
                  "subtitleColor": INK2, "subtitleFontSize": 12, "offset": 10},
        "axis": {"labelColor": INK2, "titleColor": INK2, "labelFontSize": 11,
                 "titleFontSize": 12, "titleFontWeight": 500, "gridColor": GRID,
                 "domainColor": MUTED, "tickColor": MUTED, "labelPadding": 4},
        "axisX": {"grid": False},
        "legend": {"labelColor": INK2, "titleColor": INK2, "labelFontSize": 11,
                   "titleFontSize": 11, "orient": "top", "symbolStrokeWidth": 2},
        "line": {"strokeWidth": 2},
        "point": {"size": 64, "filled": True},
        "bar": {"cornerRadiusEnd": 4},
        "range": {"category": SERIES},
    }}


def _png(spec: dict, scale: float = 2.0) -> bytes:
    import vl_convert as vlc

    return vlc.vegalite_to_png(spec, scale=scale)


def _bundle_renderer(spec: dict, **_):
    return {
        "application/vnd.vegalite.v5+json": spec,
        "image/png": base64.b64encode(_png(spec)).decode(),
        "text/plain": "<Altair chart>",
    }


def enable() -> None:
    if hasattr(alt, "theme") and hasattr(alt.theme, "register"):  # Altair >= 5.5
        alt.theme.register("actuarial", enable=True)(_theme)
    else:
        alt.themes.register("actuarial", _theme)
        alt.themes.enable("actuarial")
    alt.renderers.register("bundle", _bundle_renderer)
    alt.renderers.enable("bundle")
    alt.data_transformers.disable_max_rows()


def save(chart: alt.TopLevelMixin, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png(chart.to_dict()))


def _color(domain: list[str]) -> alt.Scale:
    return alt.Scale(domain=domain, range=SERIES[: len(domain)])


# --------------------------------------------------------------------------
# Reaseguro: notas
# --------------------------------------------------------------------------
def layer_anatomy(d: float, m: float, xmax: float) -> alt.Chart:
    x = np.linspace(0, xmax, 401)
    r = np.minimum(np.maximum(x - d, 0), m)
    df = pd.DataFrame({"x": np.r_[x, x], "y": np.r_[x - r, r],
                       "serie": ["Pérdida neta N(x)"] * len(x) + ["Recuperación R(x)"] * len(x)})
    dom = ["Pérdida neta N(x)", "Recuperación R(x)"]
    lines = alt.Chart(df).mark_line().encode(
        x=alt.X("x:Q", title="Pérdida por ocurrencia x (mdp)"),
        y=alt.Y("y:Q", title="Millones de pesos"),
        color=alt.Color("serie:N", scale=_color(dom), title=None),
        tooltip=[alt.Tooltip("x:Q", format=",.0f"), "serie:N", alt.Tooltip("y:Q", format=",.0f")],
    )
    rules = alt.Chart(pd.DataFrame({"v": [d, d + m], "t": ["d", "d + m"]})).mark_rule(
        color=MUTED, strokeDash=[4, 4]).encode(x="v:Q")
    text = alt.Chart(pd.DataFrame({"v": [d, d + m], "t": [f"d = {d:,.0f}", f"d + m = {d + m:,.0f}"]})
                     ).mark_text(align="left", dx=4, dy=-6, color=INK2, fontSize=11).encode(
        x="v:Q", y=alt.value(12), text="t:N")
    return (lines + rules + text).properties(
        width=W, height=H, title=alt.Title("Anatomía de una capa XL",
                                           subtitle=f"Prioridad d = {d:,.0f}, capacidad m = {m:,.0f}"))


# --------------------------------------------------------------------------
# Reaseguro: datos daneses
# --------------------------------------------------------------------------
def annual_counts(counts: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(counts)
    bars = base.mark_bar(color=BLUE, width=22).encode(
        x=alt.X("Year:O", title="Año", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("n:Q", title="Siniestros por año"),
        tooltip=["Year:O", alt.Tooltip("n:Q", title="observados"),
                 alt.Tooltip("fit:Q", title="Poisson con tendencia", format=".1f")])
    line = base.mark_line(color=ORANGE, point=alt.OverlayMarkDef(color=ORANGE, size=40)).encode(
        x="Year:O", y="fit:Q")
    return (bars + line).properties(width=W, height=H, title=alt.Title(
        "Frecuencia anual de incendios grandes (Dinamarca, 1980–1990)",
        subtitle="Barras: conteos observados · línea naranja: GLM de Poisson con tendencia log-lineal"))


def exceedance(emp: pd.DataFrame, fits: pd.DataFrame) -> alt.Chart:
    dom = list(dict.fromkeys(fits["modelo"]))
    pts = alt.Chart(emp).mark_circle(size=14, color=MUTED, opacity=.55).encode(
        x=alt.X("x:Q", scale=alt.Scale(type="log"), title="Pérdida (millones de DKK de 1985, escala log)"),
        y=alt.Y("S:Q", scale=alt.Scale(type="log"), title="P(X > x) (escala log)"),
        tooltip=[alt.Tooltip("x:Q", format=".2f"), alt.Tooltip("S:Q", format=".4f")])
    lines = alt.Chart(fits).mark_line().encode(
        x="x:Q", y="S:Q", color=alt.Color("modelo:N", scale=_color(dom), title=None),
        tooltip=["modelo:N", alt.Tooltip("x:Q", format=".2f"), alt.Tooltip("S:Q", format=".5f")])
    return (pts + lines).properties(width=W, height=H + 40, title=alt.Title(
        "Cola de la severidad: empírica contra modelos ajustados",
        subtitle="Puntos grises: supervivencia empírica · una cola pesada se ve como recta en log-log"))


def mean_excess(me: pd.DataFrame, u: float) -> alt.Chart:
    pts = alt.Chart(me).mark_line(color=BLUE, point=alt.OverlayMarkDef(size=20, color=BLUE)).encode(
        x=alt.X("u:Q", title="Umbral u (mDKK)"),
        y=alt.Y("e:Q", title="Exceso medio e(u) = E[X − u | X > u]"),
        tooltip=[alt.Tooltip("u:Q", format=".1f"), alt.Tooltip("e:Q", format=".2f"),
                 alt.Tooltip("n:Q", title="excesos")])
    rule = alt.Chart(pd.DataFrame({"u": [u]})).mark_rule(color=ORANGE, strokeDash=[4, 4]).encode(x="u:Q")
    return (pts + rule).properties(width=W, height=H, title=alt.Title(
        "Función de exceso medio",
        subtitle=f"Crecimiento aproximadamente lineal ⇒ cola tipo Pareto generalizada (umbral u = {u:g})"))


def cost_heatmap(grid: pd.DataFrame, best: dict) -> alt.Chart:
    feas = grid[grid["factible"]]
    heat = alt.Chart(feas).mark_rect().encode(
        x=alt.X("d:O", title="Prioridad d (mDKK)", axis=alt.Axis(labelAngle=0, values=sorted(grid.d.unique())[::3])),
        y=alt.Y("m:O", title="Capacidad m (mDKK)", sort="descending",
                axis=alt.Axis(values=sorted(grid.m.unique())[::3])),
        color=alt.Color("J:Q", scale=alt.Scale(range=SEQ[::-1]), title="Costo J"),
        tooltip=[alt.Tooltip("d:Q"), alt.Tooltip("m:Q"), alt.Tooltip("J:Q", format=",.2f"),
                 alt.Tooltip("ES_net:Q", title="ES neto", format=",.1f")])
    infeas = alt.Chart(grid[~grid["factible"]]).mark_rect(color=GRID).encode(
        x="d:O", y=alt.Y("m:O", sort="descending"),
        tooltip=[alt.Tooltip("d:Q"), alt.Tooltip("m:Q"),
                 alt.Tooltip("ES_net:Q", title="ES neto (no factible)", format=",.1f")])
    star = alt.Chart(pd.DataFrame([best])).mark_point(shape="diamond", size=180, color=ORANGE,
                                                       filled=True, stroke=SURFACE, strokeWidth=2).encode(
        x="d:O", y=alt.Y("m:O", sort="descending"),
        tooltip=[alt.Tooltip("d:Q"), alt.Tooltip("m:Q"), alt.Tooltip("J:Q", format=",.2f")])
    return (infeas + heat + star).properties(width=W, height=H + 60, title=alt.Title(
        "Costo esperado J(d, m) sobre la región factible",
        subtitle="Gris: viola ES₀.₉₉(pérdida neta anual) ≤ B · rombo: óptimo en la rejilla"))


def annual_distribution(sims: pd.DataFrame) -> alt.Chart:
    dom = ["Bruta", "Neta (capa óptima)"]
    return alt.Chart(sims).transform_density(
        "loss", groupby=["serie"], as_=["loss", "dens"], extent=[float(sims.loss.min()), float(sims.loss.quantile(.999))],
        steps=300,
    ).mark_area(opacity=.35, line={"strokeWidth": 2}).encode(
        x=alt.X("loss:Q", title="Pérdida anual (mDKK)"),
        y=alt.Y("dens:Q", title="Densidad", stack=None),
        color=alt.Color("serie:N", scale=_color(dom), title=None),
        tooltip=["serie:N", alt.Tooltip("loss:Q", format=",.0f")],
    ).properties(width=W, height=H, title=alt.Title(
        "Pérdida anual simulada: bruta contra neta de reaseguro",
        subtitle="Poisson compuesto; severidad empírica en el cuerpo y GPD en la cola"))


def frontier(df: pd.DataFrame, best: dict) -> alt.Chart:
    line = alt.Chart(df).mark_line(color=BLUE, point=alt.OverlayMarkDef(color=BLUE, size=40)).encode(
        x=alt.X("B:Q", title="Presupuesto de cola B = cota de ES₀.₉₉ de la pérdida neta anual (mDKK)",
                scale=alt.Scale(zero=False)),
        y=alt.Y("costo_transferencia:Q", title="Costo esperado de transferir: J − E S (mDKK)"),
        tooltip=[alt.Tooltip("B:Q", format=",.0f"), alt.Tooltip("ratio:Q", title="B / ES bruto", format=".0%"),
                 alt.Tooltip("d:Q", title="d óptima"), alt.Tooltip("m:Q", title="m óptima"),
                 alt.Tooltip("costo_transferencia:Q", title="costo", format=",.2f")])
    return line.properties(width=W, height=H, title=alt.Title(
        "Frontera costo–riesgo de la capa XL",
        subtitle="Cada punto: capa (d, m) de mínimo costo que cumple el presupuesto de cola"))


# --------------------------------------------------------------------------
# Supervivencia
# --------------------------------------------------------------------------
def survival_fit(df: pd.DataFrame) -> alt.Chart:
    dom = list(dict.fromkeys(df["modelo"]))
    return alt.Chart(df).mark_line(interpolate="step-after").encode(
        x=alt.X("t:Q", title="Años de desarrollo desde la ocurrencia"),
        y=alt.Y("S:Q", title="Probabilidad de seguir abierto", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("modelo:N", scale=_color(dom), title=None),
        strokeDash=alt.StrokeDash("modelo:N", scale=alt.Scale(domain=dom, range=[[1, 0], [6, 3], [2, 2]][: len(dom)]),
                                  legend=None),
        tooltip=["modelo:N", "t:Q", alt.Tooltip("S:Q", format=".3f")],
    ).properties(width=W, height=H, title=alt.Title(
        "Supervivencia de expedientes en el conjunto de prueba",
        subtitle="Kaplan–Meier observado contra el promedio de las curvas predichas (casi se superponen: buena calibración marginal)"))


def survival_profiles(df: pd.DataFrame) -> alt.Chart:
    dom = list(dict.fromkeys(df["perfil"]))
    return alt.Chart(df).mark_line(interpolate="step-after").encode(
        x=alt.X("t:Q", title="Años de desarrollo"),
        y=alt.Y("S:Q", title="Probabilidad de seguir abierto", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("perfil:N", scale=_color(dom), title=None),
        tooltip=["perfil:N", "t:Q", alt.Tooltip("S:Q", format=".3f")],
    ).properties(width=W, height=H, title=alt.Title(
        "Red neuronal: supervivencia por perfil de riesgo",
        subtitle="Siniestros reales de incendio (Francia, 1992–2007)"))


def hazard_calibration(df: pd.DataFrame) -> alt.Chart:
    diag = alt.Chart(pd.DataFrame({"a": [0, 1], "b": [0, 1]})).mark_line(color=MUTED, strokeDash=[4, 4]).encode(
        x="a:Q", y="b:Q")
    dom = list(dict.fromkeys(df["modelo"]))
    pts = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=50)).encode(
        x=alt.X("pred:Q", title="Riesgo predicho de liquidarse en el año", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("obs:Q", title="Frecuencia observada", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("modelo:N", scale=_color(dom), title=None),
        tooltip=["modelo:N", alt.Tooltip("pred:Q", format=".3f"), alt.Tooltip("obs:Q", format=".3f"),
                 alt.Tooltip("n:Q", title="persona-año")])
    return (diag + pts).properties(width=H + 60, height=H + 60, title=alt.Title(
        "Calibración del riesgo anual de liquidación",
        subtitle="Deciles de riesgo predicho, registros persona-año de prueba"))


def brier_bars(df: pd.DataFrame) -> alt.Chart:
    dom = list(dict.fromkeys(df["modelo"]))
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("horizonte:O", title="Horizonte (años)", axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("modelo:N", sort=dom),
        y=alt.Y("brier:Q", title="Brier IPCW (menor es mejor)"),
        color=alt.Color("modelo:N", scale=_color(dom), title=None, sort=dom),
        tooltip=["modelo:N", "horizonte:O", alt.Tooltip("brier:Q", format=".4f")],
    ).properties(width=W, height=H, title=alt.Title(
        "Error de predicción con censura (Brier ponderado por IPCW)",
        subtitle="Kaplan–Meier, regresión logística en tiempo discreto y red neuronal"))
