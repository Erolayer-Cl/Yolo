#!/usr/bin/env bash
# Copia px4_msgs a src/px4_msgs. Las definiciones TIENEN que coincidir con el
# firmware PX4 que corre: si no, los tópicos /fmu/out/* aparecen pero no llega
# nada (no se pueden deserializar), sin ningún error visible.
#
#   ./scripts/vendor_px4_msgs.sh release/1.17   # PX4 v1.17.x (SITL y Pixhawk): rama
#                                               # oficial de github.com/PX4/px4_msgs
#   ./scripts/vendor_px4_msgs.sh                # copia la de UAV_ros2-beta (o PX4_MSGS_SRC)
#
# Después: colcon build --packages-select px4_msgs person_yolo
# La versión copiada queda anotada en src/px4_msgs/PX4_MSGS_SOURCE.
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="$BASE/src/px4_msgs"
REF="${1:-}"

if [ -n "$REF" ]; then
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  echo "Descargando PX4/px4_msgs ($REF)..."
  git clone -q --depth 1 --branch "$REF" https://github.com/PX4/px4_msgs.git "$TMP/px4_msgs"
  SRC="$TMP/px4_msgs"
  DESC="github.com/PX4/px4_msgs $REF ($(git -C "$SRC" rev-parse --short HEAD))"
else
  SRC="${PX4_MSGS_SRC:-$BASE/../UAV_ros2-beta/src/px4_msgs}"
  if [ ! -d "$SRC/msg" ]; then
    echo "ERROR: no se encontró $SRC/msg -- ¿está inicializado el submódulo en la beta?"
    echo "       (o exportá PX4_MSGS_SRC=/ruta/a/px4_msgs, o pasá una rama: release/1.17)"
    exit 1
  fi
  DESC="$SRC ($(git -C "$SRC" describe --tags --always 2>/dev/null || echo 'sin git'))"
fi

# Se borra lo anterior: un .msg que ya no existe en la versión nueva quedaría
# compilado y con otra definición.
rm -rf "$DST"
mkdir -p "$DST"
cp -a "$SRC"/. "$DST"/
rm -rf "$DST/.git" "$DST/.github"
echo "$DESC" > "$DST/PX4_MSGS_SOURCE"

echo "px4_msgs vendorizado en $DST ($(ls "$DST"/msg | wc -l) .msg) desde $DESC"
echo "Ahora: colcon build --packages-select px4_msgs person_yolo"
