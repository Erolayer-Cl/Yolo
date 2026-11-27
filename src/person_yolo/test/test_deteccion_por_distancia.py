"""Conteo de scripts/deteccion_por_distancia.py (sin ROS, GPU ni modelo)."""
import importlib.util
from pathlib import Path

import pytest

_RUTA = Path(__file__).resolve().parents[3] / 'scripts' / 'deteccion_por_distancia.py'
_spec = importlib.util.spec_from_file_location('deteccion_por_distancia', _RUTA)
ddd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ddd)


def test_contar():
    assert ddd.contar([0, 1, 2, 0]) == {'frames': 4, 'frames_con_deteccion': 2,
                                        'tasa': 0.5, 'cajas_por_frame': 0.75}


def test_contar_sin_frames_no_divide_por_cero():
    assert ddd.contar([]) == {'frames': 0, 'frames_con_deteccion': 0,
                              'tasa': 0.0, 'cajas_por_frame': 0.0}


@pytest.mark.parametrize('nombre,esperado', [
    ('dist_h15_d10_r1', (15.0, 10.0, 'de_pie', 1)),
    ('dist_h15_d50_tendida_r2', (15.0, 50.0, 'tendida', 2)),
    ('dist_h10.5_d100_sentada_r3', (10.5, 100.0, 'sentada', 3)),
    ('dist_h15_vacia_r1', (15.0, None, 'vacia', 1)),
])
def test_datos_de_la_corrida(nombre, esperado):
    d = ddd.datos_de_la_corrida(nombre)
    assert (d['altura_m'], d['distancia_m'], d['pose'], d['repeticion']) == esperado


@pytest.mark.parametrize('nombre', ['vuelo1', 'dist_h15_d10', 'dist_d10_r1', 'dist_h15_d10_r1_x'])
def test_nombre_que_no_sigue_el_formato(nombre):
    assert ddd.datos_de_la_corrida(nombre) is None


def _fila(modelo, distancia, tasa, altura=15.0):
    return {'modelo': modelo, 'altura_m': altura, 'distancia_m': distancia, 'tasa': tasa}


def test_distancia_maxima():
    filas = [_fila('coco', 10, 0.9), _fila('coco', 25, 0.4), _fila('coco', 50, 0.1),
             _fila('n', 10, 1.0), _fila('n', 25, 0.8), _fila('n', 50, 0.5), _fila('n', 100, 0.2),
             _fila('n', None, 0.7)]                       # escena vacía: no cuenta
    assert ddd.distancia_maxima(filas) == {('coco', 15.0): 10, ('n', 15.0): 50}


def test_distancia_maxima_ninguna_y_por_altura():
    filas = [_fila('coco', 10, 0.2), _fila('coco', 10, 0.6, altura=20.0)]
    assert ddd.distancia_maxima(filas) == {('coco', 15.0): None, ('coco', 20.0): 10}
