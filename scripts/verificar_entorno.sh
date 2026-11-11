#!/bin/bash
# Revisa, sin instalar ni cambiar nada, que esté todo lo necesario para correr el
# SITL. Por cada pieza dice ✅ / ❌ (falta) / ⚠️ (funciona pero distinto a lo probado),
# y qué hacer. La guía completa: docs/INSTALACION.md.
#
#   ./scripts/verificar_entorno.sh      (o: make doctor)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UAV_LAB_DIR="${UAV_LAB_DIR:-$(cd "$REPO/../.." && pwd)}"
[ -d "$UAV_LAB_DIR/dependencias" ] || UAV_LAB_DIR="$HOME/Workspace/UAV_lab"
PX4_DIR="${PX4_DIR:-$UAV_LAB_DIR/dependencias/PX4-Autopilot}"
AIRSIM_DIR="$UAV_LAB_DIR/dependencias/Cosys-AirSim"
FALTAN=0

ok()   { echo "✅ $1"; }
mal()  { echo "❌ $1"; echo "     → $2"; FALTAN=$((FALTAN + 1)); }
aviso(){ echo "⚠️  $1"; echo "     → $2"; }

echo "Repo: $REPO"
echo "Dependencias: $UAV_LAB_DIR/dependencias   (otro lugar: UAV_LAB_DIR=... o PX4_DIR=...)"
echo

echo "— Windows / WSL (INSTALACION.md paso 1)"
if grep -qi microsoft /proc/version 2>/dev/null; then ok "corriendo en WSL ($(uname -r))"
else aviso "no parece WSL" "el SITL de este repo está probado en WSL2 con Unreal en Windows"; fi
. /etc/os-release 2>/dev/null
[ "$VERSION_ID" = "24.04" ] && ok "Ubuntu $VERSION_ID" || aviso "Ubuntu ${VERSION_ID:-?}" "probado en 24.04 (ROS 2 Jazzy)"
WSLCFG=""
if command -v cmd.exe >/dev/null 2>&1; then
  WINHOME=$(wslpath "$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')" 2>/dev/null)
  WSLCFG="$WINHOME/.wslconfig"
fi
if [ -n "$WSLCFG" ] && grep -qi '^ *networkingMode *= *mirrored' "$WSLCFG" 2>/dev/null; then
  ok "networkingMode=mirrored en $WSLCFG"
else
  mal "WSL sin networkingMode=mirrored" "agregar [wsl2] networkingMode=mirrored en %USERPROFILE%\\.wslconfig y 'wsl --shutdown' (sin esto WSL no ve a Unreal por localhost)"
fi
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then ok "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
else aviso "nvidia-smi no responde" "sin GPU el detector corre en CPU (muy lento): instalar el driver NVIDIA en Windows"; fi

echo; echo "— ROS 2 y herramientas (pasos 2 y 3)"
if [ -f /opt/ros/jazzy/setup.bash ]; then ok "ROS 2 Jazzy en /opt/ros/jazzy"
else mal "falta ROS 2 Jazzy" "ver INSTALACION.md paso 2"; fi
for p in vision-msgs foxglove-msgs foxglove-bridge cv-bridge; do
  dpkg -s "ros-jazzy-$p" >/dev/null 2>&1 && ok "ros-jazzy-$p" || mal "falta ros-jazzy-$p" "sudo apt install ros-jazzy-$p"
done
for c in colcon tmux git; do
  command -v $c >/dev/null && ok "$c" || mal "falta $c" "sudo apt install $([ $c = colcon ] && echo python3-colcon-common-extensions || echo $c)"
done
command -v MicroXRCEAgent >/dev/null && ok "MicroXRCEAgent ($(command -v MicroXRCEAgent))" \
  || mal "falta MicroXRCEAgent" "compilarlo: INSTALACION.md paso 3"

echo; echo "— PX4 y Cosys-AirSim (pasos 4 y 5)"
if [ -d "$PX4_DIR" ]; then
  v=$(git -C "$PX4_DIR" describe --tags --always 2>/dev/null)
  [ "$v" = "v1.17.0" ] && ok "PX4 $v en $PX4_DIR" || aviso "PX4 $v en $PX4_DIR" "probado con v1.17.0 (px4_msgs tiene que coincidir)"
  [ -d "$PX4_DIR/build/px4_sitl_default" ] && ok "PX4 SITL compilado" \
    || aviso "PX4 SITL sin compilar" "la primera corrida lo compila sola (~5 min): cd $PX4_DIR && make px4_sitl_default"
else mal "no existe $PX4_DIR" "INSTALACION.md paso 4"; fi
if [ -d "$AIRSIM_DIR" ]; then
  ok "Cosys-AirSim $(git -C "$AIRSIM_DIR" describe --tags --always 2>/dev/null) en $AIRSIM_DIR"
  [ -f "$AIRSIM_DIR/ros2/install/setup.bash" ] && ok "Cosys-AirSim ros2/ compilado" \
    || mal "falta compilar $AIRSIM_DIR/ros2" "INSTALACION.md paso 5"
else mal "no existe $AIRSIM_DIR" "INSTALACION.md paso 5"; fi

echo; echo "— Este repo (paso 6)"
python3 -c "import ultralytics" >/dev/null 2>&1 && ok "ultralytics $(python3 -c 'import ultralytics; print(ultralytics.__version__)')" \
  || mal "python3 no encuentra ultralytics" "pip install --user --break-system-packages ultralytics \"numpy<2\""
python3 -c "import numpy, sys; sys.exit(int(numpy.__version__.split('.')[0]) >= 2)" 2>/dev/null \
  && ok "numpy < 2 (cv_bridge de ROS lo necesita)" \
  || aviso "numpy 2 en el python del sistema" "si cv_bridge falla: pip install --user --break-system-packages \"numpy<2\""
grep -q "release/1.17" "$REPO/src/px4_msgs/PX4_MSGS_SOURCE" 2>/dev/null && ok "px4_msgs de PX4 1.17" \
  || mal "px4_msgs no es el de 1.17" "./scripts/vendor_px4_msgs.sh release/1.17"
[ -f "$REPO/install/setup.bash" ] && ok "workspace compilado" || mal "workspace sin compilar" "make build"
ls "$REPO"/models/*.pt >/dev/null 2>&1 && ok "pesos en models/: $(cd "$REPO/models" && ls *.pt | tr '\n' ' ')" \
  || aviso "no hay .pt en models/" "el de sitl.yaml (yolo11n.pt) se baja solo; los de VisDrone: training/entrenar_todo.sh"

if [ -n "$WINHOME" ]; then
  dst="$WINHOME/Documents/AirSim/settings.json"
  if [ ! -f "$dst" ]; then mal "no hay $dst" "make airsim"
  elif cmp -s "$dst" "$REPO/sim/airsim/settings.json" || cmp -s "$dst" "$REPO/sim/airsim/settings.uav_ros2.json"; then
    ok "settings.json de AirSim = el de sim/airsim/"
  else aviso "settings.json de AirSim distinto al de sim/airsim/" "make airsim (hace respaldo del actual)"; fi
fi

echo
if [ $FALTAN -eq 0 ]; then echo "Todo listo: make sitl (con Unreal en Play)."
else echo "Faltan $FALTAN cosas (❌). Guía paso a paso: docs/INSTALACION.md"; fi
exit $FALTAN
