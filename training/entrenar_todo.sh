#!/bin/bash
# Todo el entrenamiento en orden, con un solo comando. Salta lo que ya está hecho.
#
#   ./entrenar_todo.sh                 # etapas 1 2 3 5 (la 4 necesita un vuelo grabado)
#   ETAPAS="3 5" ./entrenar_todo.sh    # solo evaluar y regenerar los resultados
#   ETAPAS=5 ./entrenar_todo.sh        # solo los gráficos y la página (sin GPU)
#   ETAPAS=4 BAG=~/bags/vuelo1 ./entrenar_todo.sh   # comparar en el SITL
#
# Variables: DATA_DIR (~/datasets), ETAPAS, BAG, PY (python del venv).
# Requisitos: ver README.md de esta carpeta (venv de requirements.txt, GPU).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO=..
DATA_DIR="${DATA_DIR:-$HOME/datasets}"
DATA="$DATA_DIR/visdrone_tiles/data.yaml"
ETAPAS="${ETAPAS:-1 2 3 5}"
PY="${PY:-python3}"
[ -z "$VIRTUAL_ENV" ] && [ -f ~/venvs/yolo-train/bin/activate ] && . ~/venvs/yolo-train/bin/activate && PY=python

paso() { echo; echo "━━ $*"; }
hay() { [[ " $ETAPAS " == *" $1 "* ]]; }

if hay 1; then
  paso "Etapa 1: VisDrone -> tiles de 640 ($DATA)"
  if [ -f "$DATA" ]; then echo "ya está: se salta (borrar $DATA_DIR/visdrone_tiles para rehacerla)"
  else $PY e1_preparar_datos.py --repo $REPO --src "$DATA_DIR/VisDrone" --out "$DATA_DIR/visdrone_tiles" --download
  fi
fi

if hay 2; then
  for par in n_final:yolo11n.pt s_final:yolo11s.pt; do
    nombre=${par%%:*}; base=${par#*:}
    paso "Etapa 2: entrenar $nombre desde $base (50 épocas, patience 15)"
    if [ -f "runs/train/$nombre/weights/best.pt" ]; then echo "ya está: se salta"
    else $PY e2_entrenar.py --data "$DATA" --model "$base" --name "$nombre"
    fi
  done
fi

if hay 3; then
  paso "Etapa 3: evaluar en VisDrone test-dev (COCO vs n_final vs s_final)"
  if [ -f runs/eval/resumen.json ] && [ -z "$REHACER" ]; then
    echo "ya está runs/eval/resumen.json: se salta (REHACER=1 para repetirla)"
  else
    $PY e3_evaluar.py --repo $REPO --data "$DATA" --src "$DATA_DIR/VisDrone" \
      --model coco=yolo11n.pt \
      --model n_final=runs/train/n_final/weights/best.pt \
      --model s_final=runs/train/s_final/weights/best.pt
  fi
fi

if hay 4; then
  paso "Etapa 4: comparar en el SITL sobre ${BAG:-(falta BAG=rosbag|video|carpeta)}"
  [ -n "$BAG" ] || { echo "Definí BAG=~/bags/vuelo1 (ros2 bag record /camera)"; exit 1; }
  $PY e4_comparar_sitl.py --src "$BAG"
fi

if hay 5; then
  paso "Etapa 5: gráficos, tabla y página -> ../docs/resultados/"
  $PY e5_resultados.py
fi

# Los pesos al lugar donde los busca el detector (models/, ignorado por git).
for par in n_final:visdrone_n.pt s_final:visdrone_s.pt; do
  src="runs/train/${par%%:*}/weights/best.pt"; dst="$REPO/models/${par#*:}"
  if [ -f "$src" ] && ! cmp -s "$src" "$dst"; then cp "$src" "$dst"; echo "copiado $src -> $dst"; fi
done
echo; echo "Listo. Resultados: docs/resultados/RESULTADOS.md y resultados.html"
