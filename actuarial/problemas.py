"""Problema 7 de las notas, resuelto con datos reales.

    python -m actuarial.problemas

Para dos bases con unidad estadística identificable (incendios daneses y
ciclones tropicales mexicanos del CENAPRED):

1. Frecuencia: Poisson (MLE) contra binomial negativa; prueba del índice de
   dispersión  D = sum (n_t - nbar)^2 / nbar  ~  chi^2_{T-1}  bajo Poisson.
2. Severidad, dos modelos: (A) lognormal; (B) cuerpo empírico + GPD sobre u.
3. Validación central (cuerpo): QQ, KS con valor p por bootstrap paramétrico
   (los parámetros se estiman, así que la tabla de KS clásica no aplica).
4. Validación de cola (fuera de muestra): se ajusta en los primeros años y se
   cuentan en los últimos las excedencias de los cuantiles 90, 95 y 99%;
   prueba binomial bilateral contra lo esperado. Se compara además el exceso
   medio observado sobre u con el que implica cada modelo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import data, mexico


# ---------------------------------------------------------------------------
# Frecuencia
# ---------------------------------------------------------------------------
def frequency(counts: np.ndarray) -> dict:
    n = np.asarray(counts, float)
    T, nbar, s2 = len(n), n.mean(), n.var(ddof=1)
    D = float(np.sum((n - nbar) ** 2) / nbar)
    p = float(stats.chi2.sf(D, T - 1))
    r = nbar ** 2 / (s2 - nbar) if s2 > nbar else np.inf
    ll_pois = float(stats.poisson.logpmf(n, nbar).sum())
    ll_nb = float(stats.nbinom.logpmf(n, r, r / (r + nbar)).sum()) if np.isfinite(r) else ll_pois
    return {"años": T, "media": nbar, "varianza": s2, "var/media": s2 / nbar, "D": D, "gl": T - 1,
            "p_valor_Poisson": p, "r_binomial_negativa": r, "logver_Poisson": ll_pois, "logver_BN": ll_nb,
            "AIC_Poisson": 2 - 2 * ll_pois, "AIC_BN": 4 - 2 * ll_nb}


# ---------------------------------------------------------------------------
# Severidad
# ---------------------------------------------------------------------------
class LogNormal:
    name = "Lognormal"

    def __init__(self, x):
        self.mu, self.s = float(np.log(x).mean()), float(np.log(x).std(ddof=1))

    def cdf(self, x):
        return stats.lognorm.cdf(x, self.s, scale=np.exp(self.mu))

    def ppf(self, q):
        return stats.lognorm.ppf(q, self.s, scale=np.exp(self.mu))

    def mean_excess(self, u):
        """E[X - u | X > u] en forma cerrada."""
        z = (np.log(u) - self.mu) / self.s
        tail_mean = np.exp(self.mu + self.s ** 2 / 2) * stats.norm.sf(z - self.s)
        return float(tail_mean / stats.norm.sf(z) - u)

    def sample(self, n, rng):
        return np.exp(rng.normal(self.mu, self.s, n))


class SplicedGPD:
    name = "Empírica + GPD"

    def __init__(self, x, q_u):
        self.body = np.sort(x[x <= np.quantile(x, q_u)])
        self.u = float(np.quantile(x, q_u))
        self.p = float(np.mean(x > self.u))
        self.xi, _, self.beta = stats.genpareto.fit(x[x > self.u] - self.u, floc=0)

    def cdf(self, x):
        x = np.asarray(x, float)
        body = (1 - self.p) * np.searchsorted(self.body, x, side="right") / len(self.body)
        tail = (1 - self.p) + self.p * stats.genpareto.cdf(x - self.u, self.xi, scale=self.beta)
        return np.where(x <= self.u, body, tail)

    def ppf(self, q):
        q = np.asarray(q, float)
        qb = np.quantile(self.body, np.clip(q / (1 - self.p), 0, 1))
        qt = self.u + stats.genpareto.ppf(np.clip((q - (1 - self.p)) / self.p, 0, 1), self.xi, scale=self.beta)
        return np.where(q <= 1 - self.p, qb, qt)

    def mean_excess(self, u):
        if self.xi >= 1:
            return float("inf")  # media infinita
        return float((self.beta + self.xi * (u - self.u)) / (1 - self.xi))


def ks_bootstrap(x, model_cls, n_boot=200, seed=2026, **kw) -> tuple[float, float]:
    """KS con valor p por bootstrap paramétrico (re-estimando en cada réplica)."""
    rng = np.random.default_rng(seed)
    m = model_cls(x, **kw)
    stat = float(stats.kstest(x, m.cdf).statistic)
    if not hasattr(m, "sample"):
        return stat, float("nan")
    sims = []
    for _ in range(n_boot):
        y = m.sample(len(x), rng)
        sims.append(stats.kstest(y, model_cls(y, **kw).cdf).statistic)
    return stat, float(np.mean(np.array(sims) >= stat))


def backtest(train, test, models, levels=(.90, .95, .99)) -> pd.DataFrame:
    rows = []
    for m in models:
        for q in levels:
            xq = float(m.ppf(q))
            k = int(np.sum(test > xq))
            exp = (1 - q) * len(test)
            p = float(stats.binomtest(k, len(test), 1 - q).pvalue)
            rows.append({"modelo": m.name, "nivel": q, "cuantil ajustado": xq, "excedencias": k,
                         "esperadas": exp, "p_binomial": p})
    return pd.DataFrame(rows)


def qq_table(x, models) -> pd.DataFrame:
    xs = np.sort(x)
    q = (np.arange(1, len(xs) + 1) - .5) / len(xs)
    return pd.concat([pd.DataFrame({"modelo": m.name, "teorico": m.ppf(q), "empirico": xs}) for m in models])


def analyze(name, x, years, split_year, q_u, counts) -> dict:
    x, years = np.asarray(x, float), np.asarray(years)
    lnm, spl = LogNormal(x), SplicedGPD(x, q_u)
    ks_ln, p_ln = ks_bootstrap(x, LogNormal)
    ks_sp = float(stats.kstest(x, spl.cdf).statistic)
    tr, te = x[years < split_year], x[years >= split_year]
    models_tr = [LogNormal(tr), SplicedGPD(tr, q_u)]
    bt = backtest(tr, te, models_tr)
    u = models_tr[1].u
    me = pd.DataFrame([{"modelo": m.name, "u": u, "exceso medio del modelo": m.mean_excess(u)} for m in models_tr]
                      + [{"modelo": "Observado (prueba)", "u": u,
                          "exceso medio del modelo": float(np.mean(te[te > u] - u)) if np.any(te > u) else np.nan}])
    central = pd.DataFrame([
        {"modelo": "Lognormal", "KS": ks_ln, "p (bootstrap)": p_ln,
         "Q50 modelo": float(lnm.ppf(.5)), "Q75 modelo": float(lnm.ppf(.75))},
        {"modelo": "Empírica + GPD", "KS": ks_sp, "p (bootstrap)": np.nan,
         "Q50 modelo": float(spl.ppf(.5)), "Q75 modelo": float(spl.ppf(.75))},
    ])
    central["Q50 empírico"], central["Q75 empírico"] = np.quantile(x, .5), np.quantile(x, .75)
    return {"nombre": name, "frecuencia": frequency(counts), "central": central, "backtest": bt,
            "exceso_medio": me, "qq": qq_table(x, [lnm, spl]), "xi": spl.xi, "u": spl.u,
            "xi_train": models_tr[1].xi, "u_train": u,
            "n_train": len(tr), "n_test": len(te), "split": split_year}


def run() -> dict:
    dk = data.load_danish()
    counts_dk = dk.groupby("Year").size().to_numpy()
    out = {"dinamarca": analyze("Incendios daneses (mDKK 1985)", dk["Loss"], dk["Year"], 1988, .95, counts_dk)}
    ev = mexico.load()["eventos"]
    cy = ev[ev["tipo"] == "Ciclon tropical"]
    counts_mx = cy.groupby("anio").size().reindex(range(2000, 2025), fill_value=0).to_numpy()
    out["mexico"] = analyze("Ciclones tropicales en México (mdp 2024)", cy["dano_mdp_2024"], cy["anio"], 2017, .70,
                            counts_mx)
    return out


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:,.4f}".format)
    for k, r in run().items():
        print("\n" + "=" * 80 + f"\n{r['nombre']}  (entrenamiento < {r['split']}: n = {r['n_train']}; "
              f"prueba: n = {r['n_test']}; u = {r['u']:.2f}, xi = {r['xi']:.3f}; xi entrenamiento = {r['xi_train']:.3f})\n" + "=" * 80)
        print(pd.Series(r["frecuencia"]).to_string())
        print(r["central"].to_string(index=False))
        print(r["backtest"].to_string(index=False))
        print(r["exceso_medio"].to_string(index=False))


if __name__ == "__main__":
    main()
