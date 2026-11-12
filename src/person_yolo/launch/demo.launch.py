"""Demo de detección de personas + GPS + recorrido.

    ros2 launch person_yolo demo.launch.py                       # SITL (sitl.yaml)
    ros2 launch person_yolo demo.launch.py config:=jetson.yaml camera:=true
    ros2 launch person_yolo demo.launch.py foxglove:=false       # si ya corre otro bridge
    ros2 launch person_yolo demo.launch.py fpv:=false            # sin video FPV (:8080)
    ros2 launch person_yolo demo.launch.py model:=visdrone_s.pt  # otro peso, sin tocar el YAML

`config` acepta un nombre de archivo de config/ del paquete o una ruta completa.
`model` (vacío = el del YAML) pisa `person_detector.model`; se busca igual que en
el YAML (directorio actual, $YOLO_WS y models/).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    cfg = LaunchConfiguration('config').perform(context)
    if not os.path.isabs(cfg) and not os.path.exists(cfg):
        cfg = os.path.join(get_package_share_directory('person_yolo'), 'config', cfg)
    if not os.path.exists(cfg):
        raise RuntimeError(f'No existe el archivo de config: {cfg}')

    model = LaunchConfiguration('model').perform(context).strip()

    def node(executable, condition=None, package='person_yolo', name=None, respawn=False,
             extra=None):
        params = [cfg] + ([extra] if extra else [])     # lo último pisa al YAML
        return Node(package=package, executable=executable, name=name or executable,
                    output='screen', parameters=params, condition=condition,
                    respawn=respawn, respawn_delay=5.0)

    # Sin estos tres no hay personas en el mapa: si uno cae en vuelo, vuelve solo.
    return [
        node('camera_node', IfCondition(LaunchConfiguration('camera'))),
        node('gps_node', respawn=True),
        node('trail_node', respawn=True),
        node('person_detector', respawn=True, extra={'model': model} if model else None),
        node('fpv_node', IfCondition(LaunchConfiguration('fpv'))),
        node('foxglove_bridge', IfCondition(LaunchConfiguration('foxglove')),
             package='foxglove_bridge'),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value='sitl.yaml',
                              description='sitl.yaml | jetson.yaml | ruta a un YAML'),
        DeclareLaunchArgument('camera', default_value='false',
                              description='Lanzar camera_node (solo hardware real)'),
        DeclareLaunchArgument('foxglove', default_value='true',
                              description='Lanzar foxglove_bridge en :8765'),
        DeclareLaunchArgument('fpv', default_value='true',
                              description='Lanzar fpv_node (video FPV en http://<host>:8080)'),
        DeclareLaunchArgument('model', default_value='',
                              description='Peso de YOLO (vacío = el del YAML), '
                                          'p. ej. visdrone_n.pt'),
        OpaqueFunction(function=_nodes),
    ])
