# Cómo bajar los datos del curso

Esta guía explica, paso a paso, cómo obtener **todos** los datos que usan las notas, las soluciones de los problemas y el
notebook. Hay dos caminos: **A, automático** (un comando) y **B, manual** (desde el navegador, si la red de la escuela
bloquea las descargas por script). Ambos terminan igual: con los archivos en su lugar y verificados.

## 0. Preparación (una sola vez)

1. Instale Python 3.10 o más reciente y Git.
2. Clone el repositorio e instale las dependencias:

   ```bash
   git clone https://github.com/IvliCaesar/Actuarial_Sciences.git
   cd Actuarial_Sciences
   python -m venv .venv
   # Windows:      .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Compruebe que todo quedó instalado:

   ```bash
   python notas/optimizacion_xl.py     # debe terminar con "Validation: PASS"
   ```

> Los Problemas 1 a 6 **no necesitan datos**: se resuelven con la ley discreta de las notas
> (`python notas/soluciones_problemas.py`). Los datos sólo hacen falta para el Problema 7, el caso mexicano y el
> modelo de supervivencia.

## 1. ¿Qué datos usa cada parte?

| Parte del curso | Datos | Archivo local |
|---|---|---|
| Notas, ejemplo exacto; Problemas 1–6 | ninguno | — |
| Problema 7 (Dinamarca); sección 2 del notebook | Incendios daneses 1980–1990 | `data/raw/danishuni.rda` |
| Problema 7 (México); sección 3 del notebook | CENAPRED, desastres 2000–2024 | `data/raw_mx/cenapred.xlsx` |
| Sección 3 (seguro y reaseguro en México) | CNSF, riesgos hidrometeorológicos | `data/raw_mx/cnsf_hidro_siniestros.csv` |
| Sección 3 (sismos) | CNSF, terremoto | `data/raw_mx/cnsf_terremoto_siniestros.csv` |
| Sección 3 (pesos de 2024) | INPC de México (OCDE) | `data/raw_mx/cpi_mex.csv` |
| Sección 4 (supervivencia de siniestros) | Siniestros de incendio, Francia 1992–2007 | `data/raw/freclaimset3fire9207.rda` |

La carpeta `data/` no se sube al repositorio (está en `.gitignore`). Cada alumno la genera en su computadora.
Los **extractos mexicanos ya procesados** sí vienen en `datos_mx/`, así que el notebook corre aunque usted no baje
nada de México. Bajar las fuentes sirve para reproducir y auditar esos extractos.

## 2. Camino A: automático

```bash
python -m actuarial.descargar          # todo (~30 MB, 1 a 2 minutos)
python -m actuarial.descargar --cas    # sólo Dinamarca y Francia
python -m actuarial.descargar --mx     # sólo fuentes mexicanas
```

El programa:

* baja `danishuni.rda` y `freclaimset3fire9207.rda` de un **commit fijo** de CASdatasets y comprueba su SHA-256;
  si la huella no coincide, se detiene;
* baja las cuatro fuentes mexicanas, regenera `datos_mx/` y le dice si cada archivo es **idéntico** al que se usó en
  el repositorio o si la fuente **se actualizó** (la CNSF agrega trimestres y años; un cambio es normal y sólo
  significa que sus números diferirán un poco de los publicados).

Salida esperada (abreviada):

```
CASdatasets (commit fijo, SHA-256 verificado):
  OK  .../data/raw/danishuni.rda
  OK  .../data/raw/freclaimset3fire9207.rda
Fuentes mexicanas (CENAPRED, CNSF, INPC):
  OK  cenapred.xlsx: 2,132,864 bytes, idéntico al del repositorio
  ...
Listo.
```

Después:

```bash
python -m actuarial.problemas     # Problema 7 con datos reales
python -m actuarial               # estudio completo: figures/ y results/  (~8 min)
jupyter lab notebooks/Resultados.ipynb
```

## 3. Camino B: manual, desde el navegador

Use este camino si el comando anterior falla por proxy, certificados o bloqueos de la red. Cree las carpetas
`data/raw/` y `data/raw_mx/` dentro del repositorio, baje cada archivo y **guárdelo con el nombre exacto** de la tabla.

### 3.1 CASdatasets (Dinamarca y Francia)

| Guardar como | Enlace directo |
|---|---|
| `data/raw/danishuni.rda` | https://raw.githubusercontent.com/dutangc/CASdatasets/227fb56b8734bdb7c0327a41180e01d2ddaeaf26/data/danishuni.rda |
| `data/raw/freclaimset3fire9207.rda` | https://raw.githubusercontent.com/dutangc/CASdatasets/227fb56b8734bdb7c0327a41180e01d2ddaeaf26/data/freclaimset3fire9207.rda |

Son archivos de R (`.rda`). En Python se leen con `pyreadr`; en R, con `load("danishuni.rda")`. Documentación:
`man/danish.Rd` y `man/freclaimset9207.Rd` en https://github.com/dutangc/CASdatasets.

### 3.2 CENAPRED: base histórica de desastres 2000–2024

1. Abra el **Atlas Nacional de Riesgos, Descargas**: http://www.atlasnacionalderiesgos.gob.mx/archivo/descargas.html
2. Descargue la base del impacto socioeconómico de los desastres 2000–2024
   (enlace directo: http://www.atlasnacionalderiesgos.gob.mx/descargas/Basehistorica_2000_a_2024.xlsx).
3. Guárdela como `data/raw_mx/cenapred.xlsx`.

### 3.3 CNSF: siniestros de riesgos hidrometeorológicos y de terremoto

1. Abra el portal de **Datos Abiertos de la CNSF**: https://www.cnsf.gob.mx/Transparencia/Paginas/DatosAbiertos.aspx
2. Busque y descargue estos dos CSV:

| Archivo en el portal | Guardar como |
|---|---|
| *57 Informacion estadistica de Riesgos Hidrometeorologicos_ Siniestros.csv* (~19 MB) | `data/raw_mx/cnsf_hidro_siniestros.csv` |
| *60 Informacion estadistica de Terremoto y Erupcion Volcanica_ Siniestros.csv* (~6 MB) | `data/raw_mx/cnsf_terremoto_siniestros.csv` |

Las mismas bases aparecen en datos.gob.mx
([hidrometeorológicos](https://www.datos.gob.mx/dataset/informacion_estadistica_riesgos_hidrometeorologicos_sector_asegurador),
[terremoto](https://www.datos.gob.mx/dataset/informacion_estadistica_terremoto_erupcion_volcanica_sector_asegurador)),
que funciona desde el navegador pero **rechaza las descargas por script (error 403)**. Por eso el código usa el portal
de la CNSF. No intente disfrazar el script como navegador.

### 3.4 Índice de precios de México

1. Abra https://fred.stlouisfed.org/series/MEXCPIALLMINMEI
2. *Download → CSV*, y guarde como `data/raw_mx/cpi_mex.csv`.

(Alternativa oficial: el INPC del INEGI. Si lo usa, adapte `mexico._deflator`, porque cambian la base y el formato.)

### 3.5 Verificar y procesar

```bash
python -m actuarial.descargar --local
```

Con `--local` no se descarga nada: se verifican los archivos que usted puso y se regenera `datos_mx/`.

## 4. Verificar las huellas SHA-256 a mano

| Archivo | SHA-256 esperado |
|---|---|
| `danishuni.rda` | `cb66278bbba5236fb4ebcaef4299b10e5133963288463d396b60aa27ff2bdb4b` |
| `freclaimset3fire9207.rda` | `4d2ac53791ac47d59060edf9940a1ba76bbcf70c0357253feea123a67ec086e1` |
| `cenapred.xlsx` (versión del 13-nov-2026) | `597d14be35a0c44011a9ff7e203605e43351df38c6c5cf7e68a93234081fe10d` |
| `cnsf_hidro_siniestros.csv` (versión del 13-nov-2026) | `254caf77559a91b5e2a6fa50c11efd976812d113a6b80a35767ccc4f0dd85ecb` |
| `cnsf_terremoto_siniestros.csv` (versión del 13-nov-2026) | `b9495177d64902e66931e9ad47792741056896db17c3ea3ae8787a3db342637d` |
| `cpi_mex.csv` (versión del 13-nov-2026) | `afc67b0202566cb891f4d683e177a6ac61f9f19eb9e8d59a3862a6becb4eb34e` |

```powershell
Get-FileHash data\raw\danishuni.rda -Algorithm SHA256        # Windows (PowerShell)
```
```bash
shasum -a 256 data/raw/danishuni.rda                          # macOS
sha256sum data/raw/danishuni.rda                              # Linux
```

Las huellas de CASdatasets **deben** coincidir. Las de las fuentes mexicanas pueden cambiar cuando la institución
actualiza el archivo; `datos_mx/manifest.json` registra la versión que usó el repositorio.

## 5. Diccionario mínimo y cuidados con cada base

**`danishuni`.** `Date` (día de ocurrencia) y `Loss` (pérdida total en millones de coronas danesas, **a valores de
1985**). Sólo incluye pérdidas mayores a 1 millón: es una muestra truncada por la izquierda.

**`freclaimset3fire9207`.** Covariables `NbEmployee`, `NbSite`, `Surface`, `RiskCateg`; `OccurDate`;
`inc_Yj` y `paid_Yj`, incurrido y pagado acumulados **al cierre del año 2007 − j** (así, `Y0` es 2007 y `Y15` es 1992).
Las columnas anteriores al año de ocurrencia valen 0, no faltan.

**CENAPRED (`cenapred.xlsx`, hoja `Hoja1`).** Un renglón es un **evento en un estado**: fecha de inicio y fin, año,
clasificación (hidrometeorológico, geológico...), tipo (`Ciclón tropical`, `Sismo`, `Lluvias`...), estado,
descripción, defunciones, población afectada, viviendas y `Total de daños (millones de pesos)` en **pesos corrientes**.
Cuidados:
* el tipo aparece a veces con espacios de más (`"Ciclón tropical "`): normalice antes de filtrar;
* el daño trae textos como `SD`, `NR` o `NSR` (sin dato): conviértalos a faltante, no a cero;
* un huracán que afecta varios estados aparece en varios renglones; el código agrupa por tipo y fecha de inicio;
* es **daño económico total** (público y privado, asegurado o no; metodología DaLA de la CEPAL), no siniestro asegurado.

**CNSF (`cnsf_*_siniestros.csv`).** Codificación UTF-8 **con BOM** (en pandas use `encoding="utf-8-sig"`). Un renglón es
una **celda agregada** (año × moneda × canal × entidad × tipo de bien × cobertura × tipo de evento), no un siniestro.
Variables clave: `anio`, `entidad`, `tipo_evento` (hidrometeorológicos) o `causa_siniestro` (terremoto),
`numero_siniestros`, `monto_siniestro`, `monto_pagado`, `monto_reaseguro` (reportado desde 2021). Cuidados:
* `monto_siniestro` es el **movimiento del año** en la estimación del siniestro: puede ser **negativo** (ajustes a la baja);
* la recuperación de reaseguro de un año puede corresponder a siniestros de años anteriores (2025 incluye pagos de Otis);
* `entidad` es la del registro estadístico de la póliza, no necesariamente la del daño.

**`cpi_mex.csv`.** Índice mensual (2015 = 100); el código promedia por año y expresa todo en pesos de 2024.

## 6. Documentos públicos citados en las notas (lectura, no código)

* AMIS, comunicado sobre Otis (39 343 mdp asegurados): https://www.amisprensa.org/nota/a-un-ano-de-otis-se-han-registrado-danos-asegurados-por-39343-mdp
* AXA Seguros, *Reporte sobre la Solvencia y Condición Financiera 2025* (págs. numeradas 43 y 117): enlace en la bibliografía de las notas.
* CNSF, *Circular Única de Seguros y Fianzas*, capítulo 6.4: https://lisfcusf.cnsf.gob.mx/CUSF/CUSF6_4
* CENAPRED, *Impacto socioeconómico de los principales desastres 2024*: https://www.gob.mx/cenapred/es/articulos/impacto-socioeconomico-de-los-principales-desastres-ocurridos-en-la-republica-mexicana-2024

## 7. Problemas frecuentes

| Síntoma | Solución |
|---|---|
| `SSL: CERTIFICATE_VERIFY_FAILED` o tiempo de espera agotado | Red con proxy: use el camino B y luego `--local`. |
| `HTTP Error 403` en datos.gob.mx | Es esperado; use el portal de la CNSF (sección 3.3). |
| `ModuleNotFoundError: pyreadr` u `openpyxl` | `pip install -r requirements.txt` dentro del entorno activado. |
| `RuntimeError: SHA-256 inesperado` | El archivo `.rda` se bajó incompleto o es otro: bórrelo y vuelva a descargar. |
| Caracteres raros (`DaÃ±os`) | Leyó el CSV de la CNSF sin `encoding="utf-8-sig"`. |
| Sus números difieren un poco de los del README | La fuente mexicana se actualizó; compare su `manifest.json` con el del repositorio. |

## 8. Licencias y cómo citar

* **CASdatasets**: Dutang, C. y Charpentier, A., *CASdatasets: Insurance datasets*, licencia GPL (≥ 2).
  No se redistribuye en este repositorio; cada quien lo descarga.
* **CENAPRED** y **CNSF**: datos abiertos del Gobierno de México; cite la institución, el nombre de la base y la
  fecha de descarga (ver `datos_mx/manifest.json`).
* **INPC**: OECD, *Consumer Price Indices*, vía FRED (Federal Reserve Bank of St. Louis), serie `MEXCPIALLMINMEI`.
