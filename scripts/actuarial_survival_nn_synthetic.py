"""Discrete-time neural survival with right censoring; synthetic actuarial cash flows.
Run: python scripts/actuarial_survival_nn_synthetic.py. Requires numpy, torch, matplotlib.
No firm data; all quantities are illustrative.

Versión original (datos simulados). La versión con datos reales de siniestros
de incendio está en ``actuarial/survival.py`` y se ejecuta con ``python -m actuarial``.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 2026
K = 24
OUT = Path(__file__).resolve().parents[1] / 'figures'
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)


@dataclass
class Claims:
    features: np.ndarray
    time: np.ndarray
    event: np.ndarray
    amount: np.ndarray
    true_time: np.ndarray


def make_claims(n=3600):
    # x0 standardized log-severity predictor; x1 binary catastrophe flag
    x0 = rng.normal(size=n)
    x1 = rng.binomial(1, .35, size=n)
    X = np.column_stack((x0, x1)).astype(np.float32)
    # Time-varying monthly conditional settlement hazard.
    t = np.arange(1, K+1)
    raw = -2.7 + .085*t[None, :] + .47*x0[:, None] - .55*x1[:, None] \
        + .23*np.sin(t[None, :]/3.) * x0[:, None]
    p = 1/(1+np.exp(-raw))
    u = rng.uniform(size=p.shape)
    hit = (u < p)
    has_event = hit.any(axis=1)
    T = np.where(has_event, hit.argmax(axis=1)+1, K+1)
    # Independent administrative right censoring, conditional on observed X.
    C = rng.integers(7, K+1, size=n)
    observed = np.minimum(T, C)
    event = (T <= C).astype(np.float32)
    # Hypothetical ultimate claim amount (MXN millions), known for this toy example.
    amount = np.exp(1.0 + .52*x0 + .35*x1 + .28*rng.normal(size=n))
    return Claims(X, observed.astype('int64'), event, amount, T)


class DiscreteSurvivalNN(nn.Module):
    def __init__(self, features=2, months=K):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, 16), nn.Tanh(),
                                 nn.Linear(16, 16), nn.Tanh(),
                                 nn.Linear(16, months))

    def forward(self, x):
        return self.net(x)  # hazard logits: month 1,...,K


def censored_negative_loglik(logits, duration, event):
    # Event at t: log h_t + sum_{j<t} log(1-h_j)
    # Censored at c: sum_{j<=c} log(1-h_j)
    months = torch.arange(1, logits.shape[1]+1, device=logits.device)[None, :]
    before = (months < duration[:, None]).float()
    through = (months <= duration[:, None]).float()
    logsurv = nn.functional.logsigmoid(-logits)
    loghaz = nn.functional.logsigmoid(logits)
    event_col = (months == duration[:, None]).float()
    loglik_event = (before*logsurv + event_col*loghaz).sum(dim=1)
    loglik_cens = (through*logsurv).sum(dim=1)
    return -torch.mean(event*loglik_event + (1-event)*loglik_cens)


def survival_from_logits(logits):
    hazards = torch.sigmoid(logits)
    return torch.cat([torch.ones((len(logits), 1), device=logits.device),
                      torch.cumprod(1-hazards, axis=1)], dim=1)


def benchmark_hazard(train):
    t = np.arange(1, K+1)
    at_risk = (train.time[:, None] >= t).sum(axis=0)
    events = ((train.time[:, None] == t) * train.event[:, None]).sum(axis=0)
    return np.divide(events, at_risk, out=np.zeros(K), where=at_risk > 0)


def test_nll(model, data):
    with torch.no_grad():
        return float(censored_negative_loglik(model(torch.from_numpy(data.features)),
                     torch.from_numpy(data.time), torch.from_numpy(data.event)))


def main():
    claims = make_claims()
    perm = rng.permutation(len(claims.time))
    split = int(.75*len(perm))

    def subset(idx):
        return Claims(claims.features[idx], claims.time[idx], claims.event[idx],
                      claims.amount[idx], claims.true_time[idx])
    train, test = subset(perm[:split]), subset(perm[split:])
    model = DiscreteSurvivalNN()
    opt = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-4)
    x = torch.from_numpy(train.features)
    t = torch.from_numpy(train.time)
    e = torch.from_numpy(train.event)
    for epoch in range(200):
        model.train(); opt.zero_grad()
        loss = censored_negative_loglik(model(x), t, e)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(test.features))
        sn = survival_from_logits(logits).numpy()
    h0 = benchmark_hazard(train)
    sb = np.r_[1., np.cumprod(1-h0)]
    sbase = np.repeat(sb[None, :], len(test.time), axis=0)
    # Evaluate Brier score at 12 only on cases whose status at 12 is known.
    horizon = 12
    known = (test.time >= horizon) | ((test.event == 1) & (test.time <= horizon))
    target = (test.true_time > horizon).astype(float)
    brier_nn = np.mean((sn[known, horizon]-target[known])**2)
    brier_base = np.mean((sbase[known, horizon]-target[known])**2)
    # This is an illustrative 'complete-case Brier'; not IPCW and may be biased.
    # (La versión con datos reales usa Brier IPCW: actuarial/survival.py.)
    with torch.no_grad():
        base_logits = np.log(np.clip(h0, 1e-5, 1-1e-5)/np.clip(1-h0, 1e-5, 1-1e-5))
        baseline = torch.tensor(np.tile(base_logits, (len(test.time), 1)), dtype=torch.float32)
        base_nll = float(censored_negative_loglik(baseline, torch.from_numpy(test.time),
                                                  torch.from_numpy(test.event)))
    # Expected discount-weighted settlement disbursement if a claim is still open at a=6.
    # E[Z v^(T-a) 1{a<T<=K}|T>a,X]; this is not an IBNR reserve.
    a = 6
    v = 1/1.005
    pay_probs = np.maximum(sn[:, a:K]-sn[:, a+1:K+1], 0)/np.maximum(sn[:, [a]], 1e-10)
    pv_weights = v**np.arange(1, K-a+1)
    pv_payments = test.amount*(pay_probs@pv_weights)
    open_at_a = (test.true_time > a)
    print(f'Train/test claims: {len(train.time)}/{len(test.time)}')
    print(f'Observed settlements train: {int(train.event.sum())}')
    print(f'Neural test censored NLL: {test_nll(model, test):.4f}')
    print(f'Kaplan-Meier test censored NLL: {base_nll:.4f}')
    print(f'Illustrative observed-only Brier at month 12: NN={brier_nn:.4f}, KM={brier_base:.4f}')
    print(f'Predicted conditional discounted payout per open claim at month 6: '
          f'{np.mean(pv_payments[open_at_a]):.4f} million MXN')
    print('All results synthetic; the discounted payout excludes events after month 24.')
    # Two illustrative risk profiles, with model survival functions.
    profiles = np.array([[-1., 0.], [1., 1.]], dtype=np.float32)
    with torch.no_grad():
        sp = survival_from_logits(model(torch.from_numpy(profiles))).numpy()
    fig, ax = plt.subplots(figsize=(7, 3.7))
    for j, label in enumerate(['Perfil A', 'Perfil B']):
        ax.step(np.arange(K+1), sp[j], where='post', label=label, linewidth=1.8)
    ax.set(xlabel='Meses desde el reporte', ylabel='Probabilidad de seguir abierto',
           title='Supervivencia de expedientes: red neuronal (datos simulados)', ylim=(0, 1.02))
    ax.legend(frameon=False); ax.grid(alpha=.15); fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'sintetico_supervivencia_nn.png', dpi=180)
    print('Figure saved: figures/sintetico_supervivencia_nn.png')


if __name__ == '__main__':
    main()
