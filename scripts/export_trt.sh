#!/usr/bin/env bash
# Exporta un peso .pt a TensorRT (.engine) EN LA JETSON. Un .engine depende de
# la GPU y de la versión de TensorRT: no se copia desde otra máquina.
#
#   ./scripts/export_trt.sh models/yolo11n.pt          # imgsz 640, batch 8 dinámico
#   IMGSZ=640 BATCH=1 ./scripts/export_trt.sh models/mi_peso.pt
#
# imgsz/batch tienen que coincidir con jetson.yaml (imgsz, tiling.tile_size,
# tiling.batch_size).
set -euo pipefail
PT="${1:?uso: $0 ruta/al/peso.pt}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-8}"

DYN=False
[ "$BATCH" -gt 1 ] && DYN=True

echo "Exportando $PT -> TensorRT fp16, imgsz=$IMGSZ, batch=$BATCH (dynamic=$DYN)."
echo "Tarda varios minutos en la Jetson."
yolo export model="$PT" format=engine half=True imgsz="$IMGSZ" batch="$BATCH" \
     dynamic="$DYN" device=0
echo "Listo: ${PT%.pt}.engine  -> poné ese path en config/jetson.yaml (model)."
