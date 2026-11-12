#!/usr/bin/env bash
# Copia la configuración de Unreal/Cosys-AirSim del repo (sim/airsim/) a la carpeta
# donde la lee AirSim en Windows: %USERPROFILE%\Documents\AirSim\settings.json.
#
# Uso (desde WSL):
#   ./scripts/instalar_airsim_settings.sh            # = yolo: solo este proyecto
#   ./scripts/instalar_airsim_settings.sh uav_ros2   # este proyecto + UAV_ros2-beta
#
# Si ya hay un settings.json distinto, lo respalda antes como
# settings.json.bak-AAAAMMDD-HHMMSS. Después hay que hacer Stop/Play en Unreal.
# Otra carpeta destino: AIRSIM_DIR=/mnt/c/... ./scripts/instalar_airsim_settings.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-yolo}" in
  yolo)     SRC="$REPO/sim/airsim/settings.json" ;;
  uav_ros2) SRC="$REPO/sim/airsim/settings.uav_ros2.json" ;;
  *) echo "Uso: $0 [yolo|uav_ros2]" >&2; exit 2 ;;
esac

if [ -z "${AIRSIM_DIR:-}" ]; then
  if ! command -v cmd.exe >/dev/null || ! command -v wslpath >/dev/null; then
    echo "❌ No estoy en WSL: pasá la carpeta con AIRSIM_DIR=..." >&2
    exit 1
  fi
  WIN_HOME="$(cd /mnt/c && cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')"
  AIRSIM_DIR="$(wslpath "$WIN_HOME")/Documents/AirSim"
fi
DST="$AIRSIM_DIR/settings.json"

python3 -c "import json,sys; json.load(open(sys.argv[1], encoding='utf-8-sig'))" "$SRC" \
  || { echo "❌ $SRC no es un JSON válido" >&2; exit 1; }

mkdir -p "$AIRSIM_DIR"
if [ -f "$DST" ] && ! cmp -s "$SRC" "$DST"; then
  BAK="$DST.bak-$(date +%Y%m%d-%H%M%S)"
  cp "$DST" "$BAK"
  echo "Respaldo del anterior: $BAK"
fi
cp "$SRC" "$DST"
echo "✅ $(basename "$SRC") -> $DST"
echo "   Ahora Stop/Play en Unreal para que lo tome."
