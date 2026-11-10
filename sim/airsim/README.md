# Configuración de Unreal / Cosys-AirSim

Cosys-AirSim lee un solo archivo al apretar **Play** en Unreal:
`%USERPROFILE%\Documents\AirSim\settings.json` (en Windows). Define el dron, sus
sensores y la cámara que usa el detector. Acá está la copia versionada de ese archivo,
para que el SITL se pueda reproducir en otra máquina.

| Archivo | Cuándo usarlo |
|---|---|
| [`settings.json`](settings.json) | **Solo este proyecto.** Cámara `yolo_cam` y lidar `down`. Es la configuración con la que se midió todo lo de este repo. |
| [`settings.uav_ros2.json`](settings.uav_ros2.json) | Este proyecto **y** UAV_ros2-beta a la vez. Agrega `front_cam`, `bottom_cam` y 5 sensores de distancia horizontales. Más carga para Unreal: si la simulación se traba, usar el anterior. |

## Instalar

Desde WSL, en la raíz del repo:

```bash
./scripts/instalar_airsim_settings.sh            # settings.json (solo YOLO)
./scripts/instalar_airsim_settings.sh uav_ros2   # variante combinada
```

El script valida el JSON, respalda el `settings.json` que había como
`settings.json.bak-AAAAMMDD-HHMMSS` (solo si era distinto) y copia el nuevo. Después hay
que hacer **Stop/Play** en Unreal. A mano también sirve: copiar el archivo a
`Documents\AirSim\settings.json`.

## Qué define y dónde lo usa el código

Los valores de la cámara y del lidar **también están escritos en el código**. Si se
cambian acá, hay que cambiarlos allá. `src/person_yolo/test/test_airsim_settings.py`
lo comprueba y falla si no coinciden.

| Campo (`Vehicles.PX4…`) | Valor | Lo usa | Si se cambia |
|---|---|---|---|
| `Cameras.yolo_cam` (nombre) e `ImageType: 0` (Scene) | — | `launch_sitl.sh` remapea `/airsim_node/PX4/yolo_cam_Scene/image` → `/camera` | Cambiar el remapeo en `launch_sitl.sh` |
| `yolo_cam` `Width` × `Height` | 1920 × 1080 | Tiling del detector (`tiling.tile_size: 640`) | Con ≤ 640 px el tiling no aporta nada y las personas lejanas no se detectan |
| `yolo_cam` `FOV_Degrees` | 90 | `geolocate.camera_hfov_deg` en `config/sitl.yaml` | Cambiar el YAML, o las personas se ubican mal |
| `yolo_cam` `Pitch` | -30° | `geolocate.camera_pitch_deg` en `config/sitl.yaml`; `scripts/calibrar_geo.py` | Cambiar el YAML. Con la cámara horizontal (0°) casi ninguna persona lejana se puede ubicar en el suelo |
| `yolo_cam` `X, Y, Z` | 0.3, 0, -0.1 m | — (geolocate supone la cámara en el centro del dron) | Error de unos cm, despreciable frente al error de ubicación |
| `Sensors.down` | distancia, `Pitch -90`, 0.2–40 m, `Z 0.3` | Altura sobre el suelo para ubicar personas: `geolocate.agl_topic: /airsim_node/PX4/distance/down` | Por encima de 40 m de altura no mide y se usa la altura sobre el despegue (supone suelo plano) |
| `down.ExternalController: false` | — | AirSim lo publica en ROS pero **PX4 no lo recibe**: no cambia el vuelo | Con `true` PX4 lo usaría como telémetro y cambiaría su estimación |
| `Barometer`, `Imu`, `Gps`, `Magnetometer` | por defecto | PX4 SITL (los necesita para volar) | — |
| `OriginGeopoint` | -33.467225, -70.657605, 500 m | Lat/lon de **todo** el SITL: GPS del dron, mapa web, KML, `verdad_sitl.py` | Las coordenadas cambian; el escenario de Unreal no coincide con el mapa de calles real en ningún caso |
| `VehicleType: PX4Multirotor`, `UseTcp`, `TcpPort: 4560`, `LockStep: false` | — | Conexión con `make px4_sitl_default none_iris` (`launch_sitl.sh`) | PX4 no se conecta |
| `ControlPort: 14320`, `LocalHostIp: 0.0.0.0` | — | Puertos de AirSim para PX4, iguales que en UAV_ros2-beta | No cambiarlos sin cambiar PX4 |

Opciones que **no** están y pueden servir:
- `"ViewMode": "Fpv"` en la raíz: Unreal arranca en vista FPV (ver README §3).
- `"ClockSpeed"`: dejarlo en 1.0. Ningún nodo usa `use_sim_time` y `airsim_node` corre
  con `publish_clock:=false` (`launch_sitl.sh`): todo supone tiempo real.

## Personas en la escena

El `settings.json` no pone personas: se agregan en el editor de Unreal (MetaHumans,
maniquíes, cualquier *character* o *StaticMesh*) a distintas distancias del despegue,
por ejemplo 5, 20, 50 y 100 m. `scripts/verdad_sitl.py --listar` muestra cómo se
llaman los actores, y `--personas '<regex>'` compara sus posiciones reales con lo que
marcó el detector (ver [docs/VALIDACION.md](../../docs/VALIDACION.md)).
