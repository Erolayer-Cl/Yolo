# models/

Pesos de YOLO (no se versionan). `model:` en `config/*.yaml` busca rutas relativas en el
directorio actual y acá (`$YOLO_WS/models`, que exporta `launch_sitl.sh`).

- `.pt`: cualquier peso de Ultralytics (COCO, VisDrone, propio).

**Peso recomendado, `n_final`** (YOLO11n reentrenado con VisDrone, 09-11-2026): está en
este repo como [`visdrone_n.pt`](visdrone_n.pt) (5,5 MB). Es el único `.pt` versionado:
`.gitignore` hace una excepción para él. Para verificarlo y usarlo:

```bash
sha256sum models/visdrone_n.pt
# dda24445368c99e99ab78e4acc9aa24fca5aaae083f9da647becfd481facdb68
make sitl MODEL=visdrone_n.pt
```

Su ficha (datos, parámetros, métricas) es
[`docs/resultados/ficha_n_final.json`](../docs/resultados/ficha_n_final.json). Licencia de
los pesos: AGPL-3.0, la de Ultralytics.
- `.engine`: TensorRT, exportado **en la Jetson** con `scripts/export_trt.sh`.
  `imgsz` y `batch` del export tienen que coincidir con `jetson.yaml`.
