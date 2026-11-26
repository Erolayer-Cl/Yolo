# models/

Pesos de YOLO (no se versionan). `model:` en `config/*.yaml` busca rutas relativas en el
directorio actual y acá (`$YOLO_WS/models`, que exporta `launch_sitl.sh`).

- `.pt`: cualquier peso de Ultralytics (COCO, VisDrone, propio).

**Peso recomendado, `n_final`** (YOLO11n reentrenado con VisDrone, 09-11-2026): se publica
como `visdrone_n.pt` en la release
[`v1.0-eval02`](https://github.com/Erolayer-Cl/Yolo/releases/tag/v1.0-eval02), porque los
`.pt` no se versionan en git. Para usarlo:

```bash
gh release download v1.0-eval02 -R Erolayer-Cl/Yolo -p visdrone_n.pt -D models/
sha256sum models/visdrone_n.pt
# dda24445368c99e99ab78e4acc9aa24fca5aaae083f9da647becfd481facdb68
make sitl MODEL=visdrone_n.pt
```

Su ficha (datos, parámetros, métricas) es
[`docs/resultados/ficha_n_final.json`](../docs/resultados/ficha_n_final.json). Licencia de
los pesos: AGPL-3.0, la de Ultralytics.
- `.engine`: TensorRT, exportado **en la Jetson** con `scripts/export_trt.sh`.
  `imgsz` y `batch` del export tienen que coincidir con `jetson.yaml`.
