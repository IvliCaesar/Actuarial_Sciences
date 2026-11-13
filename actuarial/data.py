"""Descarga verificada y carga de los datos reales.

Fuente: paquete de R ``CASdatasets`` (Dutang y Charpentier), datos que acompañan
*Computational Actuarial Science with R* (CRC, 2014). Licencia GPL (>= 2).
Los archivos no se redistribuyen en este repositorio: se descargan de un commit
fijo de https://github.com/dutangc/CASdatasets y se verifica su SHA-256.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CAS_COMMIT = "227fb56b8734bdb7c0327a41180e01d2ddaeaf26"
CAS_URL = f"https://raw.githubusercontent.com/dutangc/CASdatasets/{CAS_COMMIT}/data/{{name}}.rda"
SHA256 = {
    "danishuni": "cb66278bbba5236fb4ebcaef4299b10e5133963288463d396b60aa27ff2bdb4b",
    "freclaimset3fire9207": "4d2ac53791ac47d59060edf9940a1ba76bbcf70c0357253feea123a67ec086e1",
}
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(name: str, data_dir: Path = DATA_DIR) -> Path:
    """Descarga ``name.rda`` (si no está en caché) y verifica su huella."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{name}.rda"
    if not path.exists() or _sha256(path) != SHA256[name]:
        urllib.request.urlretrieve(CAS_URL.format(name=name), path)
    digest = _sha256(path)
    if digest != SHA256[name]:
        raise RuntimeError(f"SHA-256 inesperado para {name}: {digest}")
    return path


def _read_rda(name: str) -> pd.DataFrame:
    import pyreadr  # dependencia sólo para leer .rda

    return pyreadr.read_r(str(fetch(name)))[name]


def load_danish() -> pd.DataFrame:
    """2 167 pérdidas por incendio (Copenhagen Re, 1980-1990), millones de DKK de 1985."""
    df = _read_rda("danishuni")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    return df


FIRE_FIRST, FIRE_LAST = 1992, 2007
FIRE_K = FIRE_LAST - FIRE_FIRST + 1  # 16 años de desarrollo posibles


def load_fire_claims(tol: float = 0.5) -> dict:
    """Siniestros individuales de incendio de un asegurador francés (1992-2007).

    Construye, por siniestro, el año de desarrollo de liquidación
    ``T = primer año en que pagado >= incurrido > 0`` (1 = año de ocurrencia)
    y la censura administrativa ``C = 2007 - año de ocurrencia + 1``.

    Devuelve un diccionario con el DataFrame de covariables y los arreglos
    ``inc``/``paid`` (siniestro x año calendario 1992..2007).
    """
    d = _read_rda("freclaimset3fire9207")
    occ = pd.to_datetime(d["OccurDate"]).dt.year.to_numpy()
    cal = np.arange(FIRE_FIRST, FIRE_LAST + 1)
    inc = np.stack([d[f"inc_Y{FIRE_LAST - c}"].to_numpy() for c in cal], axis=1)
    paid = np.stack([d[f"paid_Y{FIRE_LAST - c}"].to_numpy() for c in cal], axis=1)

    valid = cal[None, :] >= occ[:, None]
    settled = valid & (inc > 0) & (paid >= inc - tol)
    has = settled.any(axis=1)
    first = np.where(has, settled.argmax(axis=1), -1)
    censor = FIRE_LAST - occ + 1
    time = np.where(has, cal[first] - occ + 1, censor)

    reopened = np.zeros(len(occ), dtype=bool)
    for k in np.flatnonzero(has):
        reopened[k] = bool(np.any(paid[k, first[k]:] < inc[k, first[k]:] - tol))

    first_inc = inc[np.arange(len(occ)), occ - FIRE_FIRST]
    frame = pd.DataFrame({
        "NbEmployee": d["NbEmployee"].astype(str),
        "NbSite": d["NbSite"].astype(str),
        "Surface": d["Surface"].astype(str),
        "RiskCateg": d["RiskCateg"].astype(str),
        "occ_year": occ,
        "occ_month": pd.to_datetime(d["OccurDate"]).dt.month.to_numpy(),
        "first_incurred": first_inc,
        "latest_incurred": inc[:, -1],
        "time": time.astype("int64"),
        "event": has.astype("float32"),
        "reopened": reopened,
    })
    return {"frame": frame, "inc": inc, "paid": paid, "calendar": cal}
