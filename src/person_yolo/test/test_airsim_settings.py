"""Coherencia entre la configuración de Unreal (sim/airsim/settings*.json) y lo que
el código supone de esa cámara y ese lidar (config/sitl.yaml, launch_sitl.sh).

Si alguien cambia el FOV, el pitch o el nombre de la cámara en un lado y no en el
otro, la ubicación de las personas queda mal sin que nada avise: este test avisa.
"""
import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
SIM = REPO / 'sim' / 'airsim'
SITL_YAML = REPO / 'src' / 'person_yolo' / 'config' / 'sitl.yaml'
LAUNCH = REPO / 'launch_sitl.sh'
ARCHIVOS = ['settings.json', 'settings.uav_ros2.json']

if not SIM.is_dir():          # p. ej. en la Jetson, si solo se copió el paquete
    pytest.skip('no está sim/airsim/', allow_module_level=True)


def _settings(nombre):
    return json.loads((SIM / nombre).read_text(encoding='utf-8-sig'))


def _det():
    return yaml.safe_load(SITL_YAML.read_text())['person_detector']['ros__parameters']


@pytest.mark.parametrize('nombre', ARCHIVOS)
def test_yolo_cam_coincide_con_geolocate(nombre):
    px4 = _settings(nombre)['Vehicles']['PX4']
    cam = px4['Cameras']['yolo_cam']
    cap = cam['CaptureSettings'][0]
    geo = _det()['geolocate']
    assert cap['ImageType'] == 0                     # Scene: lo que remapea launch_sitl.sh
    assert cap['FOV_Degrees'] == pytest.approx(geo['camera_hfov_deg'])
    assert cam['Pitch'] == pytest.approx(geo['camera_pitch_deg'])


@pytest.mark.parametrize('nombre', ARCHIVOS)
def test_resolucion_alcanza_para_tiling(nombre):
    cap = _settings(nombre)['Vehicles']['PX4']['Cameras']['yolo_cam']['CaptureSettings'][0]
    tile = _det()['tiling']['tile_size']
    # Con la imagen <= tile_size el tiling no aporta nada (el detector lo avisa).
    assert max(cap['Width'], cap['Height']) > tile


@pytest.mark.parametrize('nombre', ARCHIVOS)
def test_lidar_down_es_el_que_lee_el_detector(nombre):
    down = _settings(nombre)['Vehicles']['PX4']['Sensors']['down']
    assert down['SensorType'] == 5                   # distancia
    assert down['Enabled'] and down['ExternalController'] is False   # PX4 no lo recibe
    assert down['Pitch'] == pytest.approx(-90.0)     # mira hacia abajo
    assert _det()['geolocate']['agl_topic'] == '/airsim_node/PX4/distance/down'


def test_launch_remapea_yolo_cam():
    assert '/airsim_node/PX4/yolo_cam_Scene/image:=/camera' in LAUNCH.read_text()


def test_mismo_origen_y_conexion_en_las_dos_variantes():
    a, b = (_settings(n) for n in ARCHIVOS)
    assert a['OriginGeopoint'] == b['OriginGeopoint']
    for k in ('VehicleType', 'UseTcp', 'TcpPort', 'LockStep'):
        assert a['Vehicles']['PX4'][k] == b['Vehicles']['PX4'][k]
