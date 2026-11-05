import math

import pytest

from person_yolo.geolocate import (
    Camera,
    cos_tilt,
    locate,
    offset_to_latlon,
    quat_rotate,
    touches_edge,
)

W, H = 1920, 1080
LAT, LON = -33.467225, -70.657605
LEVEL_NORTH = (1.0, 0.0, 0.0, 0.0)          # cuerpo nivelado, nariz al Norte
AGL = 10.9                                  # cámara a 10 m sobre el centro de la persona


def _yaw(deg):
    a = math.radians(deg) / 2.0
    return (math.cos(a), 0.0, 0.0, math.sin(a))  # rotación sobre z (abajo): rumbo


def _roll(deg):
    a = math.radians(deg) / 2.0
    return (math.cos(a), math.sin(a), 0.0, 0.0)


def _loc(u, v, cam, q, **kw):
    kw.setdefault('max_incidence_deg', 80.0)
    return locate(u, v, W, H, cam, q, AGL, LAT, LON, **kw)


def test_quat_rotate_yaw_90_points_east():
    n, e, d = quat_rotate(_yaw(90), (1.0, 0.0, 0.0))
    assert n == pytest.approx(0.0, abs=1e-9) and e == pytest.approx(1.0) and d == pytest.approx(0.0)


def test_nadir_center_is_below_drone():
    r = _loc(W / 2, H / 2, Camera(90, -90), LEVEL_NORTH)
    assert r['dist_m'] == pytest.approx(0.0, abs=1e-6)
    assert r['incidence_deg'] == pytest.approx(0.0, abs=1e-6)
    assert (r['lat'], r['lon']) == pytest.approx((LAT, LON))


def test_pitch_45_center_heading_north_is_10m_north():
    r = _loc(W / 2, H / 2, Camera(90, -45), LEVEL_NORTH)
    assert r['north_m'] == pytest.approx(10.0, abs=1e-6)
    assert r['east_m'] == pytest.approx(0.0, abs=1e-6)
    assert r['incidence_deg'] == pytest.approx(45.0)


def test_pitch_45_center_heading_east_is_10m_east():
    r = _loc(W / 2, H / 2, Camera(90, -45), _yaw(90))
    assert r['east_m'] == pytest.approx(10.0, abs=1e-6)
    assert r['north_m'] == pytest.approx(0.0, abs=1e-6)


def test_nadir_right_edge_is_east_when_heading_north():
    # HFOV 90: el borde derecho está a 45° -> a 10 m, 10 m al Este.
    r = _loc(W, H / 2, Camera(90, -90), LEVEL_NORTH)
    assert r['east_m'] == pytest.approx(10.0, abs=1e-6)
    assert r['north_m'] == pytest.approx(0.0, abs=1e-6)


def test_roll_right_shifts_nadir_view_left():
    # Ala derecha abajo 10°: la cámara nadir apunta hacia la izquierda (Oeste).
    r = _loc(W / 2, H / 2, Camera(90, -90), _roll(10))
    assert r['east_m'] == pytest.approx(-10.0 * math.tan(math.radians(10)), abs=1e-6)


def test_horizon_and_sky_give_none():
    assert _loc(W / 2, H / 2, Camera(90, 0), LEVEL_NORTH) is None          # horizonte
    assert _loc(W / 2, 0, Camera(90, 0), LEVEL_NORTH) is None              # cielo


def test_grazing_ray_rejected_by_default_limit():
    # Cámara a -30°: el centro llega a 60° de la vertical (pasa); el borde de
    # arriba casi en el horizonte (no pasa con el corte de 65°).
    cam = Camera(90, -30)
    assert locate(W / 2, H / 2, W, H, cam, LEVEL_NORTH, AGL, LAT, LON) is not None
    assert locate(W / 2, 0, W, H, cam, LEVEL_NORTH, AGL, LAT, LON) is None


def test_below_person_height_gives_none():
    assert locate(W / 2, H / 2, W, H, Camera(90, -90), LEVEL_NORTH, 0.5, LAT, LON) is None


def test_offset_to_latlon_scale():
    lat2, lon2 = offset_to_latlon(0.0, 0.0, 111.195, 0.0)
    assert lat2 == pytest.approx(0.001, rel=1e-3) and lon2 == 0.0


def test_error_grows_with_incidence():
    near = _loc(W / 2, H / 2, Camera(90, -90), LEVEL_NORTH)
    far = _loc(W / 2, H / 2, Camera(90, -60), LEVEL_NORTH)
    assert far['error_m'] > near['error_m'] > 0


def test_cos_tilt():
    assert cos_tilt(LEVEL_NORTH) == pytest.approx(1.0)
    assert cos_tilt(_roll(60)) == pytest.approx(0.5)
    assert cos_tilt(_yaw(73)) == pytest.approx(1.0)


@pytest.mark.parametrize('q', [
    (float('nan'), 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0, 0.0),
])
def test_invalid_attitude_gives_no_position(q):
    assert locate(W / 2, H / 2, W, H, Camera(90.0, -90.0), q, AGL, LAT, LON) is None


@pytest.mark.parametrize('agl, lat, lon', [
    (float('nan'), LAT, LON), (AGL, float('nan'), LON), (AGL, LAT, float('inf')),
])
def test_invalid_inputs_give_no_position(agl, lat, lon):
    assert locate(W / 2, H / 2, W, H, Camera(90.0, -90.0), LEVEL_NORTH, agl, lat, lon) is None


def test_touches_edge():
    assert touches_edge([0, 100, 50, 200], W, H)
    assert touches_edge([100, 1, 150, 200], W, H)
    assert touches_edge([100, 100, W - 1, 200], W, H)
    assert touches_edge([100, 100, 150, H], W, H)
    assert not touches_edge([100, 100, 150, 200], W, H)
