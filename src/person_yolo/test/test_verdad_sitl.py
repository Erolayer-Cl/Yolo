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


def test_dentro_de_2sigma_cuenta_solo_ubicados_del_detector():
    def fila(err_real, error_m=2.0, ubicada=True, manual=False):
        return {'ubicada': ubicada, 'manual': manual, 'error_m': error_m, 'err_real': err_real}
    filas = [fila(1.0), fila(3.9), fila(4.1),            # 2σ = 4 m: 2 dentro, 1 fuera
             fila(1.0, manual=True),                     # marca manual: no cuenta
             fila(1.0, error_m=None, ubicada=False),     # aproximada: no cuenta
             fila(1.0, error_m=None),                    # sin error informado: no cuenta
             fila(verdad_sitl.FALSA_ALARMA_M + 1)]       # falsa alarma: no cuenta
    assert verdad_sitl.dentro_de_2sigma(filas) == (2, 3)
    assert verdad_sitl.dentro_de_2sigma([]) == (0, 0)


def test_mediana_con_cantidad_par_promedia_los_centrales():
    # Los 9 puntos ubicados del vuelo 4 (20261111_160740), más una manual y una aproximada.
    def fila(err_real, ubicada=True, manual=False):
        return {'ubicada': ubicada, 'manual': manual, 'err_real': err_real}
    filas = [fila(e) for e in (257.0, 242.0, 145.0, 44.8, 11.4, 6.3, 7.4, 14.6, 37.4)]
    filas += [fila(1.0, manual=True), fila(1.0, ubicada=False)]
    med_cerca, max_cerca, n_cerca, med_todos, n_todos = verdad_sitl.errores_ubicados(filas)
    assert n_cerca == 4 and abs(med_cerca - 9.4) < 1e-9     # (7.4 + 11.4) / 2, no 11.4
    assert max_cerca == 14.6
    assert n_todos == 9 and med_todos == 37.4
    assert verdad_sitl.errores_ubicados([]) == (None, None, 0, None, 0)


def test_marcas_manuales_no_se_agrupan():
    filas = verdad_sitl.comparar(VERDAD, [_pt('1', 10, 't0'), _pt('M1', 10, 't0', True)])
    assert [f['real'] for f in filas] == ['B', 'B']
