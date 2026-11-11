# Evidencia de resultados

Copias de salidas reales, para que estén versionadas: `training/runs/` y
`~/yolo_flights/` no se suben a git. Cómo se leen: [`../VALIDACION.md`](../VALIDACION.md).

**Para ver las métricas de los modelos:** [`RESULTADOS.md`](RESULTADOS.md), o
`resultados.html` (doble clic; no necesita internet). Se regeneran con `make resultados`.

| Archivo | De dónde sale | Cuándo |
|---|---|---|
| `RESULTADOS.md`, `resultados.html`, `img/comparacion_metricas.png`, `img/recall_por_tamano.png`, `img/costo.png`, `img/curvas_entrenamiento.png` | `training/e5_resultados.py`: lee `runs/eval/resumen.json` y `runs/train/<run>/results.csv` y `ficha_modelo.json`; no escribe números a mano | 2026-11-11 |
| `pytest.txt` | `cd src/person_yolo && python3 -m pytest -q test` | 2026-11-10 |
| `resumen.md`, `resumen.json` | `training/e3_evaluar.py` → `training/runs/eval/` | 2026-11-09/10 |
| `ficha_n_final.json`, `ficha_s_final.json` | `training/e2_entrenar.py` → `runs/train/<run>/ficha_modelo.json`: datos, parámetros, entorno y métricas | 2026-11-09 |
| `img/ejemplo_{coco,n_final,s_final}.jpg` | `runs/eval/ejemplos_sitl_<modelo>/9999938_00000_d_0000279.jpg` (VisDrone test-dev). Verde = encontrada, rojo = no encontrada, amarillo = falsa alarma | 2026-11-10 |
| `img/curvas_*.png`, `img/pr_*.png` | Ultralytics: `runs/train/<run>/results.png` y `runs/eval/val_<run>/BoxPR_curve.png` | 2026-11-09/10 |
| `sitl/<vuelo>/` | `~/yolo_flights/<vuelo>/` (`personas.csv`, `trail.geojson`) + `scripts/verdad_sitl.py` (`verdad.csv`, `verdad.geojson`) + `informe.txt` = `verdad_sitl.py --sin-airsim --vuelo <carpeta>` | 2026-11-10 |
