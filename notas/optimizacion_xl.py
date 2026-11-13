"""Reaseguro de exceso de perdida por ocurrencia: ejemplo exactamente resoluble.

Ejecucion:  python optimizacion_xl.py      (requiere numpy y scipy)

Reproduce el Teorema de las notas: con la ley discreta
    x = (0, 500, 1500, 3128),  p = (0.70, 0.15, 0.10, 0.05),
prima Pi(d, m) = (1 + theta) E[R] + gamma m, theta = 0.25, gamma = 0.01,
y la restriccion ES_0.95(N) <= 380, la capa optima es d* = 380, m* = 2748.
La solucion certificada es la analitica; la busqueda numerica solo la valida.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution


@dataclass(frozen=True)
class ScenarioModel:
    """Distribucion discreta de una perdida por ocurrencia y tarifa de la capa."""

    losses: np.ndarray
    probs: np.ndarray
    theta: float = 0.25
    gamma: float = 0.01

    def __post_init__(self) -> None:
        x = np.asarray(self.losses, dtype=float)
        p = np.asarray(self.probs, dtype=float)
        if x.shape != p.shape or np.any(p <= 0) or not np.isclose(p.sum(), 1.0):
            raise ValueError("probabilidades invalidas")
        object.__setattr__(self, "losses", x)
        object.__setattr__(self, "probs", p)

    # --- contrato -----------------------------------------------------------
    def recovery(self, d: float, m: float) -> np.ndarray:
        """R_{d,m}(x) = min{(x - d)^+, m} en cada escenario."""
        return np.minimum(np.maximum(self.losses - d, 0.0), m)

    def net(self, d: float, m: float) -> np.ndarray:
        """N_{d,m}(x) = x - R_{d,m}(x)."""
        return self.losses - self.recovery(d, m)

    # --- momentos, prima y costo ----------------------------------------------
    def expectation(self, values: np.ndarray) -> float:
        return float(self.probs @ values)

    def premium(self, d: float, m: float) -> float:
        return (1 + self.theta) * self.expectation(self.recovery(d, m)) + self.gamma * m

    def total_cost(self, d: float, m: float) -> float:
        """J(d, m) = E[N] + Pi(d, m)."""
        return self.expectation(self.net(d, m)) + self.premium(d, m)

    # --- medida de cola -------------------------------------------------------
    def expected_shortfall(self, values: np.ndarray, alpha: float) -> float:
        """ES_alpha: promedio de la masa superior 1 - alpha, con atomos fraccionados."""
        order = np.argsort(values)[::-1]
        tail, acc = 1.0 - alpha, 0.0
        total = 0.0
        for i in order:
            w = min(self.probs[i], tail - acc)
            if w <= 0:
                break
            total += w * values[i]
            acc += w
        return total / tail

    # --- optimizacion numerica (validacion, no prueba) ------------------------
    def optimize(self, alpha: float, budget: float, seed: int = 2026):
        upper = float(self.losses.max())

        def penalized(z: np.ndarray) -> float:
            d, m = z
            excess = max(self.expected_shortfall(self.net(d, m), alpha) - budget, 0.0)
            return self.total_cost(d, m) + 1e4 * excess

        res = differential_evolution(penalized, bounds=[(0, upper), (0, upper)],
                                     seed=seed, tol=1e-10, polish=True)
        return res.x, res.fun


def main() -> None:
    model = ScenarioModel(np.array([0, 500, 1500, 3128]),
                          np.array([0.70, 0.15, 0.10, 0.05]))
    alpha, budget = 0.95, 380.0
    d, m = 380.0, 2748.0

    rec, net = model.recovery(d, m), model.net(d, m)
    er = model.expectation(rec)
    prem = model.premium(d, m)
    en = model.expectation(net)
    cost = model.total_cost(d, m)
    es = model.expected_shortfall(net, alpha)

    print(f"Gross expectation: {model.expectation(model.losses):.2f}")
    print(f"Recovery by scenario: {[int(round(r)) for r in rec]}")
    print(f"Net by scenario: {[int(round(r)) for r in net]}")
    print(f"Expected recovery: {er:.2f}")
    print(f"Premium: {prem:.2f}")
    print(f"Expected retention: {en:.2f}")
    print(f"Expected total cost: {cost:.2f}")
    print(f"ES {alpha:.0%}: {es:.2f}")

    # Identidades del Teorema y controles numericos.
    checks = [
        np.isclose(model.expectation(model.losses), 381.40),
        np.isclose(er, 267.40), np.isclose(prem, 361.73),
        np.isclose(en, 114.00), np.isclose(cost, 475.73), np.isclose(es, 380.0),
    ]
    # Rejilla finita de controles: ningun punto factible mejora el costo.
    for dd in np.linspace(0, 3128, 157):
        for mm in np.linspace(0, 3500, 176):
            if model.expected_shortfall(model.net(dd, mm), alpha) <= budget + 1e-9:
                checks.append(model.total_cost(dd, mm) >= cost - 1e-9)
    # Busqueda global con penalizacion: debe aproximar la solucion analitica.
    z, _ = model.optimize(alpha, budget)
    checks.append(abs(z[0] - d) < 1.0 and abs(z[1] - m) < 1.0)
    print("Validation:", "PASS" if all(checks) else "FAIL")


if __name__ == "__main__":
    main()
