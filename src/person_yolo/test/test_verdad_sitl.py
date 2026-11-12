"""Asignación de puntos a personas reales en scripts/verdad_sitl.py."""
import importlib.util
from pathlib import Path

_RUTA = Path(__file__).resolve().parents[3] / 'scripts' / 'verdad_sitl.py'
_spec = importlib.util.spec_from_file_location('verdad_sitl', _RUTA)
verdad_sitl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verdad_sitl)

M = 1 / 111_195        # 1 m en grados de latitud


def _pt(id_, norte_m, t, manual=False):
    return {'id': id_, 'lat': norte_m * M, 'lon': 0.0, 't': t, 'ubicada': True,
            'manual': manual, 'error_m': 2.0}


VERDAD = [('A', 0.0, 0.0), ('B', 10 * M, 0.0), ('C', 20 * M, 0.0)]


def test_vistos_juntos_van_a_personas_distintas():
    # Los dos están más cerca de B, pero se marcaron en el mismo instante:
    # son dos personas, una tiene que ser A o C.
    filas = verdad_sitl.comparar(VERDAD, [_pt('1', 9, 't0'), _pt('2', 11, 't0')])
    assert {f['real'] for f in filas} in ({'A', 'B'}, {'B', 'C'})


def test_en_momentos_distintos_se_cuenta_el_duplicado():
    filas = verdad_sitl.comparar(VERDAD, [_pt('1', 9, 't0'), _pt('2', 11, 't1')])
    assert [f['real'] for f in filas] == ['B', 'B']


def test_mas_puntos_juntos_que_personas():
    pts = [_pt(str(i), n, 't0') for i, n in enumerate([0, 10, 20, 21])]
    filas = verdad_sitl.comparar(VERDAD, pts)
    assert {f['real'] for f in filas[:3]} == {'A', 'B', 'C'}
    assert filas[3]['real'] == 'C'


def test_marcas_manuales_no_se_agrupan():
    filas = verdad_sitl.comparar(VERDAD, [_pt('1', 10, 't0'), _pt('M1', 10, 't0', True)])
    assert [f['real'] for f in filas] == ['B', 'B']
