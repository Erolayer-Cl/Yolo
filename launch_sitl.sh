#!/bin/bash
# SITL de la demo: PX4 + Unreal/Cosys-AirSim + detección de personas + GPS + recorrido.
# Adaptado de UAV_ros2-beta/launch_sim_unreal.sh, sin nodos de misión: nadie
# publica en /fmu/in/*, el dron se vuela a mano con la radio vía QGroundControl.
#
#   ./launch_sitl.sh            # todo
#   FOXGLOVE=0 ./launch_sitl.sh # sin foxglove_bridge (si ya tenés otro corriendo)
set -e
SESSION="yolo_sitl"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UAV_LAB_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
WS_DIR="$SCRIPT_DIR"
# PX4 v1.17.0, el mismo firmware que la Pixhawk real (compartido con
# UAV_ros2-beta y Mision_1).
PX4_DIR="${PX4_DIR:-$UAV_LAB_DIR/dependencias/PX4-Autopilot}"
AIRSIM_ROS_DIR="$UAV_LAB_DIR/dependencias/Cosys-AirSim/ros2"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"

if [ ! -f "$WS_DIR/install/setup.bash" ]; then
  echo "❌ Falta compilar el workspace. Corré primero:"
  echo "   ./scripts/vendor_px4_msgs.sh release/1.17 && colcon build"
  exit 1
fi
if [ ! -d "$PX4_DIR" ]; then
  echo "❌ No existe $PX4_DIR. Bajalo con:"
  echo "   git clone --branch v1.17.0 --depth 1 --recurse-submodules --shallow-submodules \\"
  echo "     https://github.com/PX4/PX4-Autopilot.git $PX4_DIR"
  exit 1
fi
# px4_msgs distinto al firmware = /fmu/out/* existen pero no llega nada.
if ! grep -q "release/1.17" "$WS_DIR/src/px4_msgs/PX4_MSGS_SOURCE" 2>/dev/null; then
  echo "⚠️  src/px4_msgs no es el de PX4 1.17. Corré:"
  echo "   ./scripts/vendor_px4_msgs.sh release/1.17 && colcon build --packages-select px4_msgs person_yolo"
fi

echo "🔧 Ajustando límite de prioridad real-time (fix glibc tpp.c de PX4)..."
# Hay que subirlo también en el SERVIDOR de tmux: los paneles son hijos suyos.
sudo prlimit --pid=$$ --rtprio=99:99 2>/dev/null || echo "⚠️  No se pudo ajustar rtprio de este script"
TMUX_SERVER_PID=$(tmux display-message -p '#{pid}' 2>/dev/null || true)
if [ -n "$TMUX_SERVER_PID" ]; then
  sudo prlimit --pid="$TMUX_SERVER_PID" --rtprio=99:99 2>/dev/null \
    || echo "⚠️  No se pudo ajustar rtprio del servidor tmux, PX4 podría fallar"
fi

echo "🧹 Limpiando procesos anteriores (también los de la beta: comparten puertos)..."
# airsim_node primero con Ctrl-C: matarlo con -9 puede crashear el plugin de AirSim en Unreal.
pkill -INT -f "airsim_node" 2>/dev/null && sleep 2 || true
tmux kill-session -t $SESSION 2>/dev/null || true
tmux kill-session -t uav_sim 2>/dev/null || true
pkill -9 -f "px4_sitl" 2>/dev/null || true
pkill -9 -f "PX4-Autopilot[^/]*/build" 2>/dev/null || true
pkill -9 -f "MicroXRCEAgent" 2>/dev/null || true
pkill -9 -f "airsim_node" 2>/dev/null || true
pkill -9 -f "foxglove_bridge" 2>/dev/null || true
pkill -9 -f "gps_bridge" 2>/dev/null || true
pkill -9 -f "yolo_detector" 2>/dev/null || true
pkill -9 -f "person_yolo/lib/person_yolo" 2>/dev/null || true
sleep 2
echo "✅ Limpieza completa."
echo ""
echo "⚠️  Antes de continuar: abrí el proyecto en Unreal Editor (Windows) y"
echo "   presioná Play. PX4 y airsim_node esperan el puerto de Cosys-AirSim."
read -p "Presioná Enter cuando Unreal esté en Play... " _

echo "🚀 Levantando SITL..."

# OJO: pane IDs (%N) capturados con -P -F, no índices: tmux renumera los
# índices en cada split y los send-keys terminarían en el panel equivocado.
PANE_PX4=$(tmux new-session -d -s $SESSION -n sim -P -F '#{pane_id}')
tmux send-keys -t "$PANE_PX4" "cd $PX4_DIR && make px4_sitl_default none_iris" C-m

PANE_AGENT=$(tmux split-window -h -t "$PANE_PX4" -P -F '#{pane_id}')
tmux send-keys -t "$PANE_AGENT" "MicroXRCEAgent udp4 -p 8888" C-m

# airsim_node: TODOS los parámetros a mano. Usa
# automatically_declare_parameters_from_overrides(true) y lee varios sobre
# variables sin inicializar si faltan -> segfault no determinista. Valores de
# airsim_node.launch.py de upstream. Ver UAV_ros2-beta/launch_sim_unreal.sh.
AIRSIM_PARAMS="-p host_ip:=localhost -p host_port:=41451 -p is_vulkan:=true -p publish_clock:=false -p enable_api_control:=false -p enable_object_transforms_list:=true -p update_airsim_control_every_n_sec:=0.01 -p update_airsim_img_response_every_n_sec:=0.033 -p update_lidar_every_n_sec:=0.01 -p update_gpulidar_every_n_sec:=0.01 -p update_echo_every_n_sec:=0.01"
# Si cambiás el nombre de la cámara o el ImageType en settings.json, cambiá esto.
AIRSIM_REMAPS="-r /airsim_node/PX4/yolo_cam_Scene/image:=/camera -r /airsim_node/PX4/yolo_cam_Scene/camera_info:=/camera_info"

PANE_AIRSIM=$(tmux split-window -v -t "$PANE_PX4" -P -F '#{pane_id}')
tmux send-keys -t "$PANE_AIRSIM" "cd $AIRSIM_ROS_DIR && source install/setup.bash && sleep 5 && ros2 run airsim_ros_pkgs airsim_node --ros-args $AIRSIM_PARAMS $AIRSIM_REMAPS" C-m

tmux select-layout -t $SESSION:sim tiled

# Ventana 2: la demo (gps_node + trail_node + person_detector + fpv_node + foxglove_bridge).
# Arranca más tarde: la carga del modelo compite por GPU con Unreal y PX4.
FOX="true"; [ "${FOXGLOVE:-1}" = "0" ] && FOX="false"
PANE_DEMO=$(tmux new-window -t $SESSION -n demo -P -F '#{pane_id}')
tmux send-keys -t "$PANE_DEMO" "cd $WS_DIR && source $ROS_SETUP && source install/setup.bash && export YOLO_WS=$WS_DIR && sleep 12 && ros2 launch person_yolo demo.launch.py config:=sitl.yaml camera:=false foxglove:=$FOX" C-m

PANE_ECHO=$(tmux split-window -v -t "$PANE_DEMO" -l 30% -P -F '#{pane_id}')
tmux send-keys -t "$PANE_ECHO" "cd $WS_DIR && source $ROS_SETUP && source install/setup.bash && sleep 15 && ros2 topic echo --no-daemon /drone/status --field data" C-m

echo ""
echo "👉 Foxglove: ws://localhost:8765   |   QGroundControl + radio USB para volar"
echo "   FPV: http://localhost:8080 (Y = cajas YOLO on/off)  |  o tecla F en Unreal"
echo "   Mapa de rescate: http://localhost:8081 (personas con sus coordenadas GPS)"
echo "   Ventana 'demo' (Ctrl-b n): logs del detector y estado del dron."
echo "   Para parar: ./stop_sitl.sh (guarda el recorrido en ~/yolo_flights/)"

tmux select-window -t $SESSION:demo
tmux attach -t $SESSION
