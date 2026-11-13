"""Reaseguro XL por ocurrencia calibrado con pérdidas reales (incendios daneses).

Pasos (cada uno corresponde a una sección de las notas):
1. Frecuencia anual: GLM de Poisson con tendencia log-lineal; proyección a 1991.
2. Severidad: Pareto, lognormal y modelo empalmado (cuerpo empírico + GPD sobre u).
3. Prima pura por capa  E R_{d,m}(X) = int_d^{d+m} S(t) dt,  contrastada con datos.
4. Simulación Poisson compuesta y optimización de (d, m) en una rejilla:
       min J(d,m) = E S + theta E R + gamma m   s.a.  ES_alpha(S_net) <= B.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Contrato y medidas de riesgo
# ---------------------------------------------------------------------------
def recovery(x: np.ndarray, d: float, m: float) -> np.ndarray:
    return np.minimum(np.maximum(x - d, 0.0), m)


def expected_shortfall(y: np.ndarray, alpha: float) -> float:
    """ES_alpha de una muestra equiprobable (masa 1 - alpha exacta en la cola)."""
    y = np.sort(np.asarray(y, dtype=float))[::-1]
    n = len(y)
    k = (1 - alpha) * n
    full = int(np.floor(k))
    return float((y[:full].sum() + (k - full) * (y[full] if full < n else 0.0)) / k)


# ---------------------------------------------------------------------------
# Frecuencia
# ---------------------------------------------------------------------------
def poisson_trend(counts: pd.DataFrame) -> dict:
    t = counts["Year"].to_numpy() - 1985.0
    n = counts["n"].to_numpy().astype(float)

    def nll(beta):
        eta = beta[0] + beta[1] * t
        return float(np.sum(np.exp(eta) - n * eta))

    beta = minimize(nll, x0=[np.log(n.mean()), 0.0], method="BFGS").x
    fit = np.exp(beta[0] + beta[1] * t)
    pearson = float(np.sum((n - fit) ** 2 / fit) / (len(n) - 2))
    return {"beta": beta, "fit": fit, "lambda_next": float(np.exp(beta[0] + beta[1] * 6.0)),
            "growth": float(np.exp(beta[1]) - 1), "dispersion": pearson,
            "lambda_mean": float(n.mean())}


# ---------------------------------------------------------------------------
# Severidad
# ---------------------------------------------------------------------------
@dataclass
class SplicedSeverity:
    """Cuerpo empírico bajo u y GPD(xi, beta) para los excesos sobre u."""

    body: np.ndarray
    u: float
    xi: float
    beta: float
    p_tail: float

    def sf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        body_sf = 1 - np.searchsorted(np.sort(self.body), x, side="right") / len(self.body)
        below = self.p_tail + (1 - self.p_tail) * body_sf
        above = self.p_tail * stats.genpareto.sf(x - self.u, self.xi, scale=self.beta)
        return np.where(x < self.u, below, above)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        tail = rng.random(n) < self.p_tail
        out = rng.choice(self.body, size=n)
        k = int(tail.sum())
        out[tail] = self.u + stats.genpareto.rvs(self.xi, scale=self.beta, size=k, random_state=rng)
        return out

    def layer_premium(self, d: float, m: float, steps: int = 4000) -> float:
        t = np.linspace(d, d + m, steps)
        return float(trapezoid(self.sf(t), t))


def fit_severity(x: np.ndarray, u: float = 10.0, n_boot: int = 300, seed: int = 2026) -> dict:
    x = np.asarray(x, dtype=float)
    xm = float(x.min())
    # Pareto de un parámetro con escala x_m = mínimo observado.
    a_par = len(x) / np.sum(np.log(x / xm))
    # Lognormal (ajuste directo a log X; la truncación en ~1 se ignora).
    mu, sig = float(np.log(x).mean()), float(np.log(x).std(ddof=1))
    # GPD sobre u.
    exc = x[x > u] - u
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    rng = np.random.default_rng(seed)
    boot = np.array([stats.genpareto.fit(rng.choice(exc, len(exc)), floc=0)[0] for _ in range(n_boot)])
    spliced = SplicedSeverity(body=x[x <= u], u=u, xi=float(xi), beta=float(beta),
                              p_tail=float(np.mean(x > u)))
    return {"pareto_alpha": float(a_par), "xm": xm, "ln_mu": mu, "ln_sigma": sig,
            "xi": float(xi), "beta": float(beta), "xi_ci": np.quantile(boot, [.025, .975]).tolist(),
            "n_exceed": int(len(exc)), "spliced": spliced}


def exceedance_tables(x: np.ndarray, fit: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    xs = np.sort(x)
    n = len(xs)
    emp = pd.DataFrame({"x": xs[:-1], "S": 1 - np.arange(1, n) / n})
    grid = np.geomspace(fit["xm"], xs.max() * 1.5, 200)
    rows = [
        ("Pareto (α = %.2f)" % fit["pareto_alpha"], (grid / fit["xm"]) ** (-fit["pareto_alpha"])),
        ("Lognormal", stats.lognorm.sf(grid, fit["ln_sigma"], scale=np.exp(fit["ln_mu"]))),
        ("Empírica + GPD (ξ = %.2f)" % fit["xi"], fit["spliced"].sf(grid)),
    ]
    fits = pd.concat([pd.DataFrame({"modelo": k, "x": grid, "S": s}) for k, s in rows])
    fits = fits[fits["S"] > 1e-5]
    return emp, fits


def mean_excess_table(x: np.ndarray, min_exceed: int = 10) -> pd.DataFrame:
    xs = np.sort(x)
    us = np.unique(np.quantile(xs, np.linspace(0, .995, 120)))
    rows = [(u, float(np.mean(xs[xs > u] - u)), int(np.sum(xs > u))) for u in us]
    df = pd.DataFrame(rows, columns=["u", "e", "n"])
    return df[df["n"] >= min_exceed]


# ---------------------------------------------------------------------------
# Simulación y optimización
# ---------------------------------------------------------------------------
@dataclass
class CompoundSample:
    year: np.ndarray       # índice de año de cada evento grande (x > d_min)
    x: np.ndarray          # severidades de esos eventos
    gross: np.ndarray      # pérdida anual bruta (todos los eventos)
    n_years: int


def simulate(sev: SplicedSeverity, lam: float, n_years: int, d_min: float, seed: int = 2026) -> CompoundSample:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam, n_years)
    x = sev.sample(int(counts.sum()), rng)
    year = np.repeat(np.arange(n_years), counts)
    gross = np.bincount(year, weights=x, minlength=n_years)
    keep = x > d_min
    return CompoundSample(year[keep], x[keep], gross, n_years)


def optimize_layer(sim: CompoundSample, d_grid, m_grid, theta: float, gamma: float,
                   alpha: float, budget: float) -> tuple[pd.DataFrame, dict]:
    ES = float(sim.gross.mean())
    rows = []
    for d in d_grid:
        for m in m_grid:
            r = np.bincount(sim.year, weights=recovery(sim.x, d, m), minlength=sim.n_years)
            er = float(r.mean())
            es_net = expected_shortfall(sim.gross - r, alpha)
            rows.append((d, m, er, ES + theta * er + gamma * m, (1 + theta) * er + gamma * m, es_net))
    grid = pd.DataFrame(rows, columns=["d", "m", "ER", "J", "prima", "ES_net"])
    grid["factible"] = grid["ES_net"] <= budget
    feas = grid[grid["factible"]]
    best = feas.loc[feas["J"].idxmin()].to_dict() if len(feas) else {}
    return grid, best


def run(danish: pd.DataFrame, theta: float = .25, gamma: float = .01, alpha: float = .99,
        budget_ratio: float = .75, u: float = 10.0, n_years: int = 50_000, seed: int = 2026) -> dict:
    x = danish["Loss"].to_numpy()
    counts = danish.groupby("Year").size().rename("n").reset_index()
    freq = poisson_trend(counts)
    counts["fit"] = freq["fit"]
    sev = fit_severity(x, u=u, seed=seed)
    emp, fits = exceedance_tables(x, sev)

    # Validación de la fórmula de prima pura por capa contra el promedio empírico.
    layers = [(5, 15), (10, 40), (20, 80), (50, 150)]
    check = pd.DataFrame([{
        "capa": f"{m:g} xs {d:g}",
        "empírica (media de R)": float(recovery(x, d, m).mean()),
        "∫ S(t) dt, modelo empalmado": sev["spliced"].layer_premium(d, m),
    } for d, m in layers])

    lam = freq["lambda_next"]
    d_grid = np.r_[np.arange(10, 100, 5.0), np.arange(100, 401, 20.0)]
    m_grid = np.r_[np.arange(0, 400, 25.0), np.arange(400, 1201, 100.0)]
    sim = simulate(sev["spliced"], lam, n_years, d_min=float(d_grid.min()), seed=seed)
    es_gross = expected_shortfall(sim.gross, alpha)
    budget = budget_ratio * es_gross
    grid, best = optimize_layer(sim, d_grid, m_grid, theta, gamma, alpha, budget)

    # Frontera: costo de transferencia mínimo para cada presupuesto de cola B.
    frontier = []
    for r in np.linspace(.55, .98, 24):
        f = grid[grid["ES_net"] <= r * es_gross]
        if len(f):
            b = f.loc[f["J"].idxmin()]
            frontier.append({"ratio": r, "B": r * es_gross, "costo_transferencia": b["J"] - sim.gross.mean(),
                             "d": b["d"], "m": b["m"]})
    frontier = pd.DataFrame(frontier)

    r_best = np.bincount(sim.year, weights=recovery(sim.x, best["d"], best["m"]), minlength=n_years)
    net = sim.gross - r_best
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_years, 4_000, replace=False)
    dist = pd.concat([pd.DataFrame({"serie": "Bruta", "loss": sim.gross[idx].round(1)}),
                      pd.DataFrame({"serie": "Neta (capa óptima)", "loss": net[idx].round(1)})])
    summary = {
        "n_siniestros": int(len(x)), "años": f"{counts.Year.min()}–{counts.Year.max()}",
        "lambda_media": freq["lambda_mean"], "lambda_1991": lam,
        "crecimiento_anual": freq["growth"], "sobredispersion_pearson": freq["dispersion"],
        "pareto_alpha": sev["pareto_alpha"], "gpd_u": u, "gpd_xi": sev["xi"], "gpd_xi_ic95": sev["xi_ci"],
        "gpd_beta": sev["beta"], "excesos_sobre_u": sev["n_exceed"],
        "E_S_bruta": float(sim.gross.mean()), f"ES_{alpha}_bruta": es_gross, "B": budget,
        "d_opt": best["d"], "m_opt": best["m"], "J_opt": best["J"], "prima_opt": best["prima"],
        "ER_opt": best["ER"], f"ES_{alpha}_neta_opt": best["ES_net"],
        "theta": theta, "gamma": gamma, "alpha": alpha, "años_simulados": n_years,
    }
    return {"counts": counts, "emp": emp, "fits": fits, "mean_excess": mean_excess_table(x),
            "layer_check": check, "grid": grid, "best": best, "dist": dist, "summary": summary,
            "frontier": frontier}
