#!/bin/bash
# Para todo lo que levanta launch_sitl.sh. trail_node guarda el recorrido al
# recibir Ctrl-C; por eso primero se manda SIGINT al launch y recién después -9.
SESSION="yolo_sitl"

echo "💾 Cerrando nodos ROS (trail_node guarda el recorrido)..."
pkill -INT -f "person_yolo demo.launch.py" 2>/dev/null || true
# airsim_node también con Ctrl-C: si se lo mata con -9, el plugin de AirSim
# en Unreal puede crashear al cortarse la conexión RPC (rpc::async_writer).
pkill -INT -f "airsim_node" 2>/dev/null || true
sleep 3

echo "🧹 Parando la simulación..."
tmux kill-session -t $SESSION 2>/dev/null || true
pkill -9 -f "px4_sitl" 2>/dev/null || true
pkill -9 -f "PX4-Autopilot[^/]*/build" 2>/dev/null || true
pkill -9 -f "MicroXRCEAgent" 2>/dev/null || true
pkill -9 -f "airsim_node" 2>/dev/null || true
pkill -9 -f "foxglove_bridge" 2>/dev/null || true
pkill -9 -f "person_yolo/lib/person_yolo" 2>/dev/null || true
echo "✅ Todo parado. Recorridos guardados en ~/yolo_flights/"
