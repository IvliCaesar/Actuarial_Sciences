"""Verificacion numerica de las soluciones de los Problemas 1-6 de las notas.

Ejecucion:  python soluciones_problemas.py      (requiere numpy)

Cada bloque reproduce una cuenta de Soluciones_Problemas.tex. Las soluciones
son analiticas; este programa solo comprueba la aritmetica y explora
rejillas finas como control independiente. El Problema 7 usa datos reales y
se resuelve en actuarial/problemas.py (python -m actuarial.problemas).
"""
from __future__ import annotations

import itertools

import numpy as np

X = np.array([0.0, 500.0, 1500.0, 3128.0])
P = np.array([0.70, 0.15, 0.10, 0.05])
THETA, GAMMA, ALPHA = 0.25, 0.01, 0.95
EX = float(P @ X)


def rec(x, d, m):
    return np.minimum(np.maximum(np.asarray(x, float) - d, 0.0), m)


def es(values, probs, alpha):
    """ES_alpha de una ley discreta: masa superior 1 - alpha, con atomos fraccionados."""
    order = np.argsort(values)[::-1]
    tail, acc, tot = 1 - alpha, 0.0, 0.0
    for i in order:
        w = min(probs[i], tail - acc)
        if w <= 1e-15:
            break
        tot += w * values[i]
        acc += w
    return tot / tail


def premium(d, m):
    return (1 + THETA) * float(P @ rec(X, d, m)) + GAMMA * m


def cost(d, m):
    return EX + THETA * float(P @ rec(X, d, m)) + GAMMA * m


def survival_integral(d, m, n=200001):
    t = np.linspace(d, d + m, n)
    s = np.array([P[X > ti].sum() for ti in t])
    return float(np.sum((s[1:] + s[:-1]) / 2 * np.diff(t)))


def header(k):
    print("\n" + "=" * 70 + f"\nProblema {k}\n" + "=" * 70)


def problema1():
    header(1)
    d, m = 500.0, 1500.0
    r, n = rec(X, d, m), X - rec(X, d, m)
    print("Recuperacion por escenario:", r.tolist())
    print("Retencion por escenario:   ", n.tolist())
    print(f"E R = {P @ r:.2f}   (integral de supervivencia: {survival_integral(d, m):.2f})")
    print(f"E N = {P @ n:.2f}   Prima = {premium(d, m):.2f}   J = {cost(d, m):.2f}")
    print(f"ES_0.95(N) = {es(n, P, ALPHA):.2f}")
    print(f"Exceso sobre la capa (x = 3128): {3128 - (d + m):.0f}")


def problema2():
    header(2)
    # Suma finita sobre intervalos de constancia de la supervivencia.
    for d, m in [(380, 2748), (500, 1500), (0, 5000), (1000, 100)]:
        pts = np.unique(np.r_[0.0, X])
        total = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            s = P[X > a].sum()                     # S constante en [a, b)
            total += s * max(0.0, min(b, d + m) - max(a, d))
        print(f"(d, m) = ({d}, {m}):  sum_j S(a_j)|I_j cap [d, d+m]| = {total:.2f}"
              f"   E R = {P @ rec(X, d, m):.2f}")


def problema3():
    header(3)
    print("B      d*     m*     E R      Prima    J        ES(N)   dJ/dB")
    for B in [250.0, 380.0, 600.0]:
        d, m = B, 3128 - B
        n = X - rec(X, d, m)
        eps = 1e-3
        slope = (cost(B + eps, 3128 - B - eps) - cost(B - eps, 3128 - B + eps)) / (2 * eps)
        print(f"{B:<6.0f} {d:<6.0f} {m:<6.0f} {P @ rec(X, d, m):<8.2f} {premium(d, m):<8.2f}"
              f" {cost(d, m):<8.2f} {es(n, P, ALPHA):<7.2f} {slope:.4f}")
        # Control por rejilla: ningun punto factible mejora el costo.
        best = min((cost(dd, mm), dd, mm) for dd in np.arange(0, 3129, 4.0)
                   for mm in np.arange(0, 3201, 4.0)
                   if es(X - rec(X, dd, mm), P, ALPHA) <= B + 1e-9)
        print(f"       rejilla: J = {best[0]:.2f} en (d, m) = ({best[1]:.0f}, {best[2]:.0f})")


def problema4():
    header(4)
    print(f"Prima minima con ES <= 380: Pi(380, 2748) = {premium(380, 2748):.2f} > 350")
    feas = [premium(dd, mm) for dd in np.arange(0, 381, 2.0) for mm in np.arange(2748, 3201, 2.0)]
    print(f"rejilla de la region ES-factible: min Pi = {min(feas):.2f}")
    b = (508.03 - 350) / 0.385
    print(f"Menor cota alcanzable con Pi <= 350: B = {b:.2f}; Pi(B, 3128 - B) = {premium(b, 3128 - b):.2f}")


def problema5():
    header(5)
    d, m = 380.0, 2748.0
    occ, agg, lim0, rp = 0.0, 0.0, 0.0, 0.0
    rows = []
    for (i, j) in itertools.product(range(4), repeat=2):
        p = P[i] * P[j]
        r1, r2 = rec(X[i], d, m), rec(X[j], d, m)
        s = X[i] + X[j]
        occ += p * (r1 + r2)
        lim0 += p * min(r1 + r2, m)
        agg += p * rec(s, d, m)
        rp += p * r1 / m
        rows.append((X[i], X[j], p, r1 + r2, min(r1 + r2, m), float(rec(s, d, m))))
    print(f"E R occ (reinstalacion ilimitada = una reinstalacion con K = 2): {occ:.2f}")
    print(f"E R occ sin reinstalacion (limite anual m): {lim0:.2f}")
    print(f"E R agregado (prioridad anual 380, limite 2748): {agg:.2f}")
    print(f"Fraccion esperada de capacidad reinstalada E[R(X1)]/m = {rp:.6f}")
    p0 = (1 + THETA) * occ / (1 + rp)
    print(f"Prima inicial de equilibrio P0 = {p0:.2f}; prima de reinstalacion esperada = {p0 * rp:.2f}")
    print(f"Comparacion: (1 + theta) E R sin reinstalacion = {(1 + THETA) * lim0:.2f}")
    # ES de la retencion anual bajo los tres contratos.
    probs = np.array([r[2] for r in rows])
    gross = np.array([r[0] + r[1] for r in rows])
    for k, name in [(3, "occ ilimitado"), (4, "occ sin reinst."), (5, "agregado")]:
        net = gross - np.array([r[k] for r in rows])
        print(f"ES_0.95 retencion anual, {name}: {es(net, probs, ALPHA):.2f}")
    print(f"ES_0.95 perdida bruta anual: {es(gross, probs, ALPHA):.2f}")


def problema6():
    header(6)
    n = X - rec(X, 380, 2748)
    print(f"ES_0.95(N) = {es(n, P, .95):.2f}   ES_0.995(N) = {es(n, P, .995):.2f}")
    print(f"ES_0.95(X) = {es(X, P, .95):.2f}   ES_0.995(X) = {es(X, P, .995):.2f}")
    # Refinamiento del atomo superior con la misma masa y la misma media.
    X2 = np.array([0, 500, 1500, 3000, 4280.0])
    P2 = np.array([.70, .15, .10, .045, .005])
    print(f"Ley refinada: E X = {P2 @ X2:.2f}, ES_0.95 = {es(X2, P2, .95):.2f}, ES_0.995 = {es(X2, P2, .995):.2f}")
    n2 = X2 - rec(X2, 380, 2748)
    print(f"Con la capa (380, 2748): ES_0.95(N) = {es(n2, P2, .95):.2f}, ES_0.995(N) = {es(n2, P2, .995):.2f}")


if __name__ == "__main__":
    print(f"E X = {EX:.2f}")
    for f in (problema1, problema2, problema3, problema4, problema5, problema6):
        f()
