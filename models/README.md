# models/

Pesos de YOLO (no se versionan). `model:` en `config/*.yaml` busca rutas relativas en el
directorio actual y acá (`$YOLO_WS/models`, que exporta `launch_sitl.sh`).

- `.pt`: cualquier peso de Ultralytics (COCO, VisDrone, propio).
- `.engine`: TensorRT, exportado **en la Jetson** con `scripts/export_trt.sh`.
  `imgsz` y `batch` del export tienen que coincidir con `jetson.yaml`.
