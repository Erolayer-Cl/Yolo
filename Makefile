# Atajos del proyecto. `make` o `make help` lista todo.
# Corren en WSL (Ubuntu 24.04 + ROS 2 Jazzy). Instalación: docs/INSTALACION.md
SHELL := /bin/bash
ROS := source /opt/ros/jazzy/setup.bash
MODEL ?=

.DEFAULT_GOAL := help
.PHONY: help doctor build test airsim sitl stop verdad informe-vuelo entrenar resultados

help: ## esta lista
	@grep -E '^[a-z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'
	@echo "  Variables: MODEL=visdrone_n.pt (make sitl), ETAPAS=\"3 5\" (make entrenar)"

doctor: ## revisa que esté todo instalado (no cambia nada)
	@./scripts/verificar_entorno.sh

build: ## compila el workspace (px4_msgs + person_yolo)
	@grep -q release/1.17 src/px4_msgs/PX4_MSGS_SOURCE 2>/dev/null || ./scripts/vendor_px4_msgs.sh release/1.17
	$(ROS) && colcon build --symlink-install

test: ## tests de la lógica (sin ROS ni GPU)
	cd src/person_yolo && python3 -m pytest -q test

airsim: ## copia sim/airsim/settings.json a Documents\AirSim (con respaldo)
	./scripts/instalar_airsim_settings.sh

sitl: ## levanta PX4 + AirSim + detector + mapa (Unreal en Play). MODEL=visdrone_n.pt
	MODEL=$(MODEL) ./launch_sitl.sh

stop: ## para el SITL y guarda el recorrido en ~/yolo_flights/
	./stop_sitl.sh

verdad: ## con Unreal en Play: posición real de las personas vs. el último vuelo
	python3 scripts/verdad_sitl.py

informe-vuelo: ## lo mismo sin Unreal, desde el verdad.csv ya guardado del último vuelo
	python3 scripts/verdad_sitl.py --sin-airsim

entrenar: ## entrenamiento completo con VisDrone (training/entrenar_todo.sh)
	ETAPAS="$(or $(ETAPAS),1 2 3 5)" ./training/entrenar_todo.sh

resultados: ## gráficos + RESULTADOS.md + resultados.html de los modelos
	ETAPAS=5 ./training/entrenar_todo.sh
