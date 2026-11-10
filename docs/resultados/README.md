# Evidencia de resultados

Copias de salidas reales, para que estén versionadas: `training/runs/` y
`~/yolo_flights/` no se suben a git. Cómo se leen: [`../VALIDACION.md`](../VALIDACION.md).

| Archivo | De dónde sale | Cuándo |
|---|---|---|
| `pytest.txt` | `cd src/person_yolo && python3 -m pytest -q test` | 2026-11-10 |
| `resumen.md`, `resumen.json` | `training/evaluate.py` → `training/runs/eval/` | 2026-11-09/10 |
| `ficha_n_final.json`, `ficha_s_final.json` | `training/train.py` → `runs/train/<run>/ficha_modelo.json`: datos, parámetros, entorno y métricas | 2026-11-09 |
| `img/ejemplo_{coco,n_final,s_final}.jpg` | `runs/eval/ejemplos_sitl_<modelo>/9999938_00000_d_0000279.jpg` (VisDrone test-dev). Verde = encontrada, rojo = no encontrada, amarillo = falsa alarma | 2026-11-10 |
| `img/curvas_*.png`, `img/pr_*.png` | Ultralytics: `runs/train/<run>/results.png` y `runs/eval/val_<run>/BoxPR_curve.png` | 2026-11-09/10 |
| `sitl/<vuelo>/` | `~/yolo_flights/<vuelo>/` (`personas.csv`, `trail.geojson`) + `scripts/verdad_sitl.py` (`verdad.csv`, `verdad.geojson`) + `informe.txt` = `verdad_sitl.py --sin-airsim --vuelo <carpeta>` | 2026-11-10 |
