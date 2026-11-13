"""Supervivencia neuronal en tiempo discreto con censura a la derecha.

Adaptación del modelo sintético original (``scripts/actuarial_survival_nn_synthetic.py``)
a siniestros reales de incendio: el "evento" es la liquidación del expediente
(pagado = incurrido) y la censura es administrativa (fin de 2007).

Verosimilitud por siniestro, con riesgos h_t = sigmoid(logit_t):
    evento en t:      log h_t + sum_{j<t} log(1 - h_j)
    censura en c:     sum_{j<=c} log(1 - h_j)

Modelos comparados: Kaplan–Meier (sin covariables), regresión logística en
tiempo discreto (red sin capa oculta) y red neuronal (dos capas ocultas).
La evaluación usa el Brier con ponderación IPCW (Graf et al., 1999).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

CATS = ["NbEmployee", "NbSite", "Surface", "RiskCateg"]


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
@dataclass
class SurvData:
    X: np.ndarray
    time: np.ndarray
    event: np.ndarray
    frame: pd.DataFrame

    def subset(self, idx: np.ndarray) -> "SurvData":
        return SurvData(self.X[idx], self.time[idx], self.event[idx], self.frame.iloc[idx])


def design_matrix(frame: pd.DataFrame, stats: dict | None = None) -> tuple[np.ndarray, list[str], dict]:
    f = frame.copy()
    f["log_first_inc"] = np.log1p(f["first_incurred"])
    f["late_report"] = (f["first_incurred"] <= 0).astype(float)
    f["msin"] = np.sin(2 * np.pi * f["occ_month"] / 12)
    f["mcos"] = np.cos(2 * np.pi * f["occ_month"] / 12)
    num = ["log_first_inc", "occ_year"]
    if stats is None:
        stats = {c: (float(f[c].mean()), float(f[c].std())) for c in num}
    for c in num:
        f[c] = (f[c] - stats[c][0]) / stats[c][1]
    dummies = pd.get_dummies(f[CATS], prefix=CATS, dtype=float)
    X = pd.concat([f[num + ["late_report", "msin", "mcos"]], dummies], axis=1)
    return X.to_numpy(np.float32), list(X.columns), stats


# ---------------------------------------------------------------------------
# Modelo y verosimilitud
# ---------------------------------------------------------------------------
class DiscreteSurvivalNN(nn.Module):
    def __init__(self, features: int, periods: int, hidden: int = 32, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        width = features
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.Tanh()]
            width = hidden
        layers.append(nn.Linear(width, periods))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # logits de riesgo para los periodos 1..K


def censored_negative_loglik(logits, duration, event):
    periods = torch.arange(1, logits.shape[1] + 1, device=logits.device)[None, :]
    before = (periods < duration[:, None]).float()
    through = (periods <= duration[:, None]).float()
    at = (periods == duration[:, None]).float()
    logsurv = nn.functional.logsigmoid(-logits)
    loghaz = nn.functional.logsigmoid(logits)
    ll_event = (before * logsurv + at * loghaz).sum(dim=1)
    ll_cens = (through * logsurv).sum(dim=1)
    return -torch.mean(event * ll_event + (1 - event) * ll_cens)


def survival_from_hazards(h: np.ndarray) -> np.ndarray:
    """S(0..K) a partir de riesgos h(1..K)."""
    return np.concatenate([np.ones((len(h), 1)), np.cumprod(1 - h, axis=1)], axis=1)


def km_hazard(time: np.ndarray, event: np.ndarray, K: int) -> np.ndarray:
    t = np.arange(1, K + 1)
    at_risk = (time[:, None] >= t).sum(axis=0)
    events = ((time[:, None] == t) * event[:, None]).sum(axis=0)
    return np.divide(events, at_risk, out=np.zeros(K), where=at_risk > 0)


def fit_model(train: SurvData, valid: SurvData, K: int, hidden: int, depth: int,
              epochs: int = 2000, lr: float = .01, wd: float = 1e-4, seed: int = 2026):
    torch.manual_seed(seed)
    model = DiscreteSurvivalNN(train.X.shape[1], K, hidden, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    xt, tt, et = (torch.from_numpy(a) for a in (train.X, train.time, train.event))
    xv, tv, ev = (torch.from_numpy(a) for a in (valid.X, valid.time, valid.event))
    best, best_state, history = np.inf, None, []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = censored_negative_loglik(model(xt), tt, et)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(censored_negative_loglik(model(xv), tv, ev))
        history.append((epoch, float(loss.detach()), vl))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history, columns=["epoch", "train", "valid"])


def predict_hazards(model, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy().astype(float)


def nll_from_hazards(h: np.ndarray, d: SurvData) -> float:
    logits = torch.from_numpy(np.log(np.clip(h, 1e-6, 1 - 1e-6) / np.clip(1 - h, 1e-6, 1)))
    return float(censored_negative_loglik(logits, torch.from_numpy(d.time), torch.from_numpy(d.event)))


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def censoring_survival(time: np.ndarray, event: np.ndarray, K: int) -> np.ndarray:
    """Distribución de la censura por Kaplan–Meier inverso.

    Censura al final del periodo c: el siniestro se observó durante 1..c.
    Un evento en t no informa sobre la censura después de t.
    """
    t = np.arange(1, K + 1)
    at_risk = (time[:, None] >= t).sum(axis=0)
    cens = ((time[:, None] == t) * (1 - event)[:, None]).sum(axis=0)
    q = np.divide(cens, at_risk, out=np.zeros(K), where=at_risk > 0)
    return np.r_[1.0, np.cumprod(1 - q)]  # G[j] = P(C > j) = P(C >= j + 1)


def ipcw_brier(S: np.ndarray, d: SurvData, h: int, G: np.ndarray) -> float:
    """Brier IPCW de P(T > h | X) en el horizonte h.

    Estado conocido si T <= h con evento, o si el siniestro se observó durante
    todo 1..h sin liquidarse. G[j] estima P(C >= j + 1), es decir, P(C > j).
    """
    settled = (d.event == 1) & (d.time <= h)
    open_known = d.time >= h
    open_known &= ~settled
    w = np.zeros(len(d.time))
    # Evento en T: se requiere C >= T  ->  P(C > T - 1) = G[T - 1].
    w[settled] = 1 / np.maximum(G[d.time[settled] - 1], 1e-8)
    # Abierto conocido en h: se requiere C >= h -> G[h - 1].
    w[open_known] = 1 / max(G[h - 1], 1e-8)
    target = open_known.astype(float)
    return float(np.sum(w * (S[:, h] - target) ** 2) / len(d.time))


def person_period_calibration(h: np.ndarray, d: SurvData, bins: int = 10) -> pd.DataFrame:
    K = h.shape[1]
    t = np.arange(1, K + 1)
    risk = d.time[:, None] >= t
    y = (d.time[:, None] == t) & (d.event[:, None] == 1)
    p, o = h[risk], y[risk].astype(float)
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    g = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    df = pd.DataFrame({"g": g, "p": p, "o": o}).groupby("g").agg(pred=("p", "mean"), obs=("o", "mean"),
                                                                n=("o", "size")).reset_index(drop=True)
    return df


def km_curve(time: np.ndarray, event: np.ndarray, K: int) -> np.ndarray:
    return survival_from_hazards(km_hazard(time, event, K)[None, :])[0]


# ---------------------------------------------------------------------------
# Flujo completo
# ---------------------------------------------------------------------------
def run(fire: dict, K: int = 16, seed: int = 2026, rate: float = .03, a: int = 2) -> dict:
    torch.set_num_threads(4)
    frame = fire["frame"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(frame))
    n_tr, n_va = int(.6 * len(perm)), int(.15 * len(perm))
    idx_tr, idx_va, idx_te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    _, cols, st = design_matrix(frame.iloc[idx_tr])
    X, cols2, _ = design_matrix(frame, st)
    assert cols == cols2 or set(cols) <= set(cols2)
    full = SurvData(X, frame["time"].to_numpy(), frame["event"].to_numpy(np.float32), frame)
    tr, va, te = full.subset(idx_tr), full.subset(idx_va), full.subset(idx_te)

    nn_model, hist = fit_model(tr, va, K, hidden=32, depth=2, seed=seed)
    glm_model, _ = fit_model(tr, va, K, hidden=0, depth=0, seed=seed)

    h_nn, h_glm = predict_hazards(nn_model, te.X), predict_hazards(glm_model, te.X)
    h_km = np.tile(km_hazard(tr.time, tr.event, K), (len(te.time), 1))
    hz = {"Kaplan–Meier": h_km, "Logística (GLM)": h_glm, "Red neuronal": h_nn}
    S = {k: survival_from_hazards(v) for k, v in hz.items()}

    G = censoring_survival(te.time, te.event, K)
    horizons = [1, 2, 3, 5, 8]
    brier = pd.DataFrame([{"modelo": k, "horizonte": hh, "brier": ipcw_brier(S[k], te, hh, G)}
                          for k in hz for hh in horizons])
    metrics = pd.DataFrame([{"modelo": k, "NLL prueba (por siniestro)": nll_from_hazards(v, te)}
                            for k, v in hz.items()])
    ibs = brier.groupby("modelo", sort=False)["brier"].mean().rename("Brier IPCW medio").reset_index()
    metrics = metrics.merge(ibs, on="modelo")

    t = np.arange(K + 1)
    km_te = km_curve(te.time, te.event, K)
    curves = pd.concat([pd.DataFrame({"modelo": "Kaplan–Meier observado (prueba)", "t": t, "S": km_te}),
                        pd.DataFrame({"modelo": "Logística (GLM), promedio", "t": t, "S": S["Logística (GLM)"].mean(0)}),
                        pd.DataFrame({"modelo": "Red neuronal, promedio", "t": t, "S": S["Red neuronal"].mean(0)})])

    calib = pd.concat([person_period_calibration(h_glm, te).assign(modelo="Logística (GLM)"),
                       person_period_calibration(h_nn, te).assign(modelo="Red neuronal")])

    # Perfiles: siniestro "típico" del conjunto de prueba variando el tamaño inicial.
    base = te.frame.iloc[[0]].copy()
    base["occ_year"], base["occ_month"] = 2000, 6
    for c, v in {"NbEmployee": "(0,10]", "NbSite": "(0,10]", "Surface": "(0,10]", "RiskCateg": "C1"}.items():
        base[c] = v
    q = np.quantile(frame.loc[frame.first_incurred > 0, "first_incurred"], [.25, .75, .99])
    profiles = pd.concat([base.assign(first_incurred=v) for v in q])
    # Se concatenan con todos los datos para que las variables indicadoras coincidan.
    Xp_df = pd.DataFrame(design_matrix(pd.concat([frame, profiles]), st)[0][-3:], columns=cols2)
    Sp = survival_from_hazards(predict_hazards(nn_model, Xp_df.to_numpy(np.float32)))
    labels = [f"Incurrido inicial p25 (€{q[0]:,.0f})", f"p75 (€{q[1]:,.0f})", f"p99 (€{q[2]:,.0f})"]
    prof = pd.concat([pd.DataFrame({"perfil": lab, "t": t, "S": Sp[j]}) for j, lab in enumerate(labels)])

    # Reserva de casos descontada: siniestros de prueba abiertos al cierre del año a.
    inc, paid = fire["inc"][idx_te], fire["paid"][idx_te]
    occ = te.frame["occ_year"].to_numpy()
    col = occ - 1992 + a - 1
    open_a = (te.time > a) & (col <= inc.shape[1] - 1)
    rows = np.flatnonzero(open_a)
    outstanding = np.maximum(inc[rows, col[rows]] - paid[rows, col[rows]], 0)
    Sn = S["Red neuronal"][rows]
    cond = (Sn[:, a:K] - Sn[:, a + 1:K + 1]) / np.maximum(Sn[:, [a]], 1e-10)
    v = (1 + rate) ** -np.arange(1, K - a + 1)
    pv_factor = cond @ v
    tail_mass = Sn[:, K] / np.maximum(Sn[:, a], 1e-10)
    exp_years = cond @ np.arange(1, K - a + 1)
    reserve = {
        "a": a, "tasa": rate, "siniestros_abiertos": int(len(rows)),
        "reserva_casos": float(outstanding.sum()),
        "reserva_descontada": float(np.sum(outstanding * pv_factor)),
        "factor_descuento_medio_ponderado": float(np.sum(outstanding * pv_factor) / max(outstanding.sum(), 1)),
        "años_esperados_a_liquidar_medio": float(np.mean(exp_years)),
        "masa_no_liquidada_en_K": float(np.mean(tail_mass)),
    }

    fr = frame
    summary = {
        "n_siniestros": int(len(fr)), "liquidados": int(fr.event.sum()),
        "censurados": int((1 - fr.event).sum()), "reabiertos_tras_liquidar": int(fr.reopened.sum()),
        "n_entrenamiento": int(len(idx_tr)), "n_validación": int(len(idx_va)), "n_prueba": int(len(idx_te)),
        "épocas_mejor_validación": int(hist.loc[hist.valid.idxmin(), "epoch"]),
    }
    return {"metrics": metrics, "brier": brier, "curves": curves, "calibration": calib,
            "profiles": prof, "reserve": reserve, "summary": summary, "history": hist}
