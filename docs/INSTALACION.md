# Instalación desde cero (Windows + WSL)

Desde una PC con Windows sin nada instalado hasta ver personas en el mapa de rescate.
Cada paso termina con una **comprobación**: si no da lo esperado, no seguir.

En cualquier momento, `./scripts/verificar_entorno.sh` (o `make doctor`) revisa todas
las piezas sin cambiar nada, y dice qué falta y cómo arreglarlo.

## Lo que se va a instalar

Son las versiones con las que se probó todo (relevadas en la PC del equipo el 2026-11-11).
Con otras versiones puede funcionar, pero no está probado.

| Pieza | Versión probada | Dónde corre |
|---|---|---|
| Windows | 11 (10.0.26100) | — |
| Driver NVIDIA | 610.88 (RTX 4060) | Windows; WSL lo usa |
| WSL | 2.7.10, `networkingMode=mirrored` | Windows |
| Ubuntu | 24.04.4 LTS | WSL |
| ROS 2 | Jazzy | WSL |
| Micro XRCE-DDS Agent | 2.4.2 | WSL |
| PX4-Autopilot | v1.17.0 | WSL |
| Unreal Engine | 5.8 | Windows |
| Cosys-AirSim | plugin 3.5.0 precompilado (Windows); `ros2/` del tag 5.8-v3.4.1 (WSL) | los dos |
| Proyecto de Unreal | `Pruebas` (mapa `World_Demo`), no versionado: ver paso 5a | Windows |
| Python del detector | 3.12.3, `ultralytics` 8.4.123, torch 2.13.0, numpy < 2 | WSL |
| QGroundControl | la última estable | Windows |

Lo que necesita la PC: GPU NVIDIA (Unreal + YOLO a la vez), ~100 GB libres (Unreal se
lleva la mayor parte) y 16 GB de RAM o más. La del equipo tiene una RTX 4060 de 8 GB y le
da 24 GB a WSL.

Layout de carpetas que esperan los scripts (si se usa otro, ver el paso 6):

```
~/Workspace/UAV_lab/
├── dependencias/
│   ├── PX4-Autopilot/        (paso 4)
│   └── Cosys-AirSim/         (paso 5)
└── Code/
    └── YOLO/                 este repo (paso 6)
```

---

## 1. Windows: driver, WSL y red

1. Instalar el **driver NVIDIA** para Windows (el normal, de nvidia.com). En WSL **no** se
   instala ningún driver: WSL usa el de Windows.
2. En PowerShell **como administrador**:
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
   Reiniciar si lo pide. Al abrir "Ubuntu 24.04" la primera vez, crear el usuario de Linux.
3. Crear (o editar) `C:\Users\<tu usuario>\.wslconfig`:
   ```ini
   [wsl2]
   networkingMode=mirrored
   guiApplications=true
   memory=24GB        # ajustar a la RAM de la PC (dejar ~8 GB para Windows y Unreal)
   processors=16      # ajustar a los núcleos de la PC
   ```
   **`networkingMode=mirrored` es obligatorio.** Con eso WSL y Windows comparten
   `localhost`: `airsim_node` se conecta a Unreal con `host_ip:=localhost` y PX4 encuentra
   a AirSim en el puerto TCP 4560 sin configurar IPs: el `settings.json` del repo escucha
   con `"LocalHostIp": "0.0.0.0"`. Sin mirrored habría que poner ahí la IP de WSL (lo
   explica `Cosys-AirSim/docs/px4_sitl_wsl2.md`). Este repo no está probado de esa forma.
4. Aplicarlo: `wsl --shutdown` en PowerShell y volver a abrir Ubuntu.

**Comprobación** (dentro de Ubuntu):
```bash
cat /etc/os-release | grep VERSION=      # 24.04
nvidia-smi                               # muestra la GPU
```
Y en PowerShell, `wsl --version` muestra la versión 2.x.

## 2. ROS 2 Jazzy y paquetes

Se siguen los pasos oficiales: <https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html>.
Si algo de acá deja de funcionar, manda esa página.

```bash
sudo apt update && sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
sudo apt update && sudo apt upgrade -y

# ROS 2 + lo que usa este proyecto
sudo apt install -y ros-jazzy-desktop ros-dev-tools \
    ros-jazzy-vision-msgs ros-jazzy-foxglove-msgs ros-jazzy-foxglove-bridge ros-jazzy-cv-bridge \
    python3-colcon-common-extensions python3-pip python3-pytest python3-rosdep \
    tmux git cmake build-essential
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && source ~/.bashrc
```

**Comprobación:** `ros2 --help` responde, y `ros2 pkg list | grep -E "vision_msgs|foxglove"`
muestra 3 paquetes.

## 3. Micro XRCE-DDS Agent (puente PX4 ↔ ROS 2)

PX4 publica sus tópicos `/fmu/out/*` a ROS 2 a través de este agente.

```bash
cd ~ && git clone -b v2.4.2 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent && mkdir -p build && cd build
cmake .. && make -j$(nproc) && sudo make install && sudo ldconfig /usr/local/lib/
```

**Comprobación:** `MicroXRCEAgent --help` muestra `Usage: 'MicroXRCEAgent <udp4|...`.

## 4. PX4-Autopilot v1.17.0

```bash
mkdir -p ~/Workspace/UAV_lab/dependencias && cd ~/Workspace/UAV_lab/dependencias
git clone --branch v1.17.0 --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot
bash ./Tools/setup/ubuntu.sh --no-nuttx     # dependencias del SITL (no las de la placa)
```
Cerrar y volver a abrir la terminal (el script agrega grupos y variables). Después:
```bash
cd ~/Workspace/UAV_lab/dependencias/PX4-Autopilot
make px4_sitl_default                     # primera compilación: ~5-10 min
```

**Comprobación:** `git describe --tags` da `v1.17.0`, y existe `build/px4_sitl_default/bin/px4`.
No hace falta correrlo ahora: `launch_sitl.sh` lo arranca con el modelo `none_iris`, que
espera a AirSim.

## 5. Unreal Engine 5.8 + Cosys-AirSim 5.8-v3.4.1

### 5a. En Windows: Unreal y el plugin

1. Instalar el **Epic Games Launcher** (<https://www.unrealengine.com/download>), y desde la
   pestaña *Unreal Engine* instalar **5.8.x**.
2. Bajar de <https://github.com/Cosys-Lab/Cosys-AirSim/releases> la release para UE 5.8.
   El equipo usa el plugin **3.5.0** (`AirSim_plugin_Windows_58_350`). La release trae
   dos cosas:
   - El **plugin precompilado** para UE 5.8 en Windows.
   - El proyecto de ejemplo **Blocks**.

   Es lo más simple: compilarlo desde el código fuente requiere Visual Studio y está en
   `docs/install_windows.md` de Cosys-AirSim.
3. Descomprimir Blocks, crear `Blocks/Plugins/` y poner adentro la carpeta `AirSim` del
   plugin. Doble clic en `Blocks.uproject`: compila el proyecto y abre Unreal.
   (Para usar el plugin en otro proyecto: `docs/unreal_custenv.md` de Cosys-AirSim.)
4. **El proyecto que usa el equipo es `Pruebas`**
   (`Documents\Unreal Projects\Pruebas\Pruebas.uproject`). Lo que tiene:

   | | |
   |---|---|
   | Motor | UE 5.8 (`"EngineAssociation": "5.8"`) |
   | Plugin | `Plugins\AirSim_plugin_Windows_58_350\AirSim`: Cosys-AirSim **3.5.0** precompilado para UE 5.8.0, el zip `AirSim_plugin_Windows_58_350` de las releases |
   | Mapa | **`World_Demo`**, del pack gratuito *FREE Stylized Forest Sample* de StyleHex (`Content/StyleHex_Studio/FREE_Stylized_Forest_Sample/Levels/World_Demo`). Es el de todos los vuelos de `docs/resultados/sitl/`. El proyecto también tiene `Untitled` y `Demo` (Vefects *Free Fire*), que no se usan |
   | Personas | Agregadas a mano en el mapa (ver [`sim/airsim/README.md`](../sim/airsim/README.md)) |

   **No está en git.** Ocupa 6,2 GB y los packs de escena son de Fab, con su licencia de
   uso: no se pueden redistribuir. Para armar uno equivalente en otra PC:
   1. Crear un proyecto vacío de UE 5.8.
   2. Crearle la carpeta `Plugins/` y poner adentro la carpeta `AirSim` del zip del
      plugin, como con Blocks.
   3. Agregar el pack *FREE Stylized Forest Sample* desde Fab (gratis, con cuenta de Epic).
   4. Abrir `World_Demo`.
   5. En *World Settings*, poner como GameMode el de AirSim (`AirSimGameMode`), según
      `docs/unreal_custenv.md` de Cosys-AirSim.
   6. Poner personas a 5, 20, 50 y 100 m del despegue.

   Si alcanza con algo más simple, Blocks con personas agregadas funciona igual. Lo único
   que el código necesita del mapa es el `settings.json` de `sim/airsim/`.

   > Versiones: el plugin de Windows es la **3.5.0** y el `ros2/` de WSL (paso 5b) está
   > compilado del tag **5.8-v3.4.1**. Así funciona en la PC del equipo, pero si
   > `airsim_node` falla al conectarse, lo primero es usar la misma versión de los dos
   > lados.
5. En Unreal: *Edit → Editor Preferences → Performance* → desmarcar **Use Less CPU when
   in Background**. Si no, Unreal se pone lento cuando QGroundControl tiene el foco.

### 5b. En WSL: el código de Cosys-AirSim y su `ros2/`

`airsim_node` (el puente AirSim → ROS 2, que publica la cámara y el lidar) se compila
desde el mismo tag:

```bash
cd ~/Workspace/UAV_lab/dependencias
git clone --branch 5.8-v3.4.1 --depth 1 https://github.com/Cosys-Lab/Cosys-AirSim.git
cd Cosys-AirSim
./setup.sh && ./build.sh            # compila AirLib (lo usa el nodo de ROS 2)
cd ros2
sudo rosdep init 2>/dev/null; rosdep update
rosdep install --from-paths src -y --ignore-src --skip-keys pcl --skip-keys message_runtime --skip-keys message_generation
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```
Los pasos salen de `docs/install_linux.md` y `docs/ros2.md` del mismo tag. ✏️ Si
`build.sh` pide `--ue-root`, seguir `install_linux.md`: el equipo tiene que confirmar
cómo lo compiló.

**Comprobación:** existe `~/Workspace/UAV_lab/dependencias/Cosys-AirSim/ros2/install/setup.bash`
y `AirLib/lib/libAirLib.a`.

## 6. Este repo

```bash
mkdir -p ~/Workspace/UAV_lab/Code && cd ~/Workspace/UAV_lab/Code
git clone https://github.com/Erolayer-Cl/Yolo.git YOLO
cd YOLO

# Python del detector (el del sistema, el mismo que usa ROS)
pip install --user --break-system-packages ultralytics "numpy<2"
#   numpy < 2: cv_bridge de ROS no funciona con numpy 2 (que trae ultralytics)

make build      # px4_msgs de PX4 1.17 + person_yolo (~3 min la primera vez)
make test       # tests de la lógica: tienen que pasar todos
make airsim     # copia sim/airsim/settings.json a Documents\AirSim (con respaldo)
```

**Si el repo está en otro lugar** (por ejemplo `~/yolo-ref`), `launch_sitl.sh` prueba
`~/Workspace/UAV_lab` para encontrar las dependencias. Si están en otro lado:
`UAV_LAB_DIR=/ruta/a/UAV_lab make sitl` (o `PX4_DIR=...` solo para PX4).

**Pesos del modelo:**
- `sitl.yaml` usa `yolo11n.pt` (COCO), que Ultralytics baja solo la primera vez.
- Los reentrenados con VisDrone (`visdrone_n.pt` y `visdrone_s.pt`, recomendados) no
  están en git. Se generan con `make entrenar` (ver
  [`training/README.md`](../training/README.md)), o se copian a `models/` desde quien los
  tenga.
- Para usarlos: `make sitl MODEL=visdrone_n.pt`.

**Comprobación:** `make doctor` da todo ✅.

## 7. QGroundControl y la radio

Instalar QGroundControl para Windows
(<https://docs.qgroundcontrol.com/master/en/qgc-user-guide/getting_started/download_and_install.html>).
Cómo conectar la radio como joystick y configurar PX4 está en el
[README §3](../README.md#3-volar-con-la-radio-en-sitl).

## 8. Primera corrida

1. Abrir el proyecto en Unreal y apretar **Play**.
2. En WSL: `make sitl MODEL=visdrone_n.pt`. Se abre tmux con PX4, el agente, AirSim y el
   detector. Cuando lo pida, apretar Enter.
3. Abrir QGroundControl, esperar ~15 s, armar y despegar en *Position*.
4. En el navegador de Windows:
   - <http://localhost:8081>: mapa de rescate.
   - <http://localhost:8080>: FPV con las cajas.
5. Volar sobre las personas (20–40 m de altura).
6. `make stop`: guarda el vuelo en `~/yolo_flights/<fecha_hora>/`.
7. Con Unreal todavía en Play, `make verdad` compara lo detectado con la posición real de
   cada persona en Unreal.

**Comprobación:** en `:8081` aparecen puntos rojos (personas ubicadas) y en
`~/yolo_flights/` está el `personas.csv`.

## 9. Si algo falla

| Síntoma | Causa y solución |
|---|---|
| `airsim_node` no conecta / PX4 se queda en "Waiting for simulator" | Unreal no está en Play, o WSL no está en `mirrored` (paso 1). Revisar que el firewall de Windows permita TCP 4560 |
| PX4 se cae al arrancar con error de prioridad (`tpp.c`) | `launch_sitl.sh` sube `rtprio` con `sudo prlimit`: correrlo con un usuario con sudo |
| `cv_bridge`: `_ARRAY_API not found` | numpy 2: `pip install --user --break-system-packages "numpy<2"` |
| Detector a pocos fps | Sin GPU en WSL (`nvidia-smi`), o Unreal se come la GPU: bajar la calidad gráfica de Unreal |
| `git` desde Windows sobre `\\wsl.localhost\...`: "dubious ownership" | Usar git desde WSL, o `git -c safe.directory=* ...` |
| Otro | [README, Problemas comunes](../README.md#problemas-comunes) y `make doctor` |
