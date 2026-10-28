import json
import math
import xml.etree.ElementTree as ET

import pytest

from person_yolo.trail import Trail, TrailPoint, haversine_m

LAT0, LON0 = -33.4489, -70.6693
M_PER_DEG_LAT = 111_195.0


def pt(t, north_m=0.0, east_m=0.0, alt=10.0):
    lat = LAT0 + north_m / M_PER_DEG_LAT
    lon = LON0 + east_m / (M_PER_DEG_LAT * math.cos(math.radians(LAT0)))
    return TrailPoint(t, lat, lon, 520.0 + alt, alt, 0.0)


def test_haversine_one_degree_lat():
    assert haversine_m(0, 0, 1, 0) == pytest.approx(M_PER_DEG_LAT, rel=1e-3)


def test_decimation_ignores_hover_and_counts_distance():
    tr = Trail(min_step_m=1.0)
    assert tr.add(pt(0))
    for i in range(50):                       # hover: ruido de 10 cm
        assert not tr.add(pt(0.1 * i, north_m=0.1))
    for i in range(1, 11):                    # 10 m al norte en pasos de 1.5 m
        tr.add(pt(10 + i, north_m=1.5 * i))
    assert len(tr.points) == 11
    assert tr.total_m == pytest.approx(15.0, abs=0.05)
    assert tr.home.t == 0


def test_vertical_climb_is_recorded():
    tr = Trail(min_step_m=1.0)
    tr.add(pt(0, alt=0))
    assert tr.add(pt(1, alt=5))


def test_thinning_keeps_first_and_last():
    tr = Trail(min_step_m=0.5, max_points=100)
    for i in range(500):
        tr.add(pt(i, east_m=i))
    assert len(tr.points) <= 100
    assert tr.points[0].t == 0 and tr.points[-1].t == 499


def test_detections_merge_near_in_space_and_time():
    tr = Trail(det_merge_m=5.0, det_merge_sec=10.0)
    a = pt(0)
    for k in range(20):                       # 2 s a 10 Hz sobre la misma persona
        tr.add_detection(k * 0.1, a.lat, a.lon, 10.0, 1 + (k == 5), 0.5 + 0.01 * k)
    assert len(tr.detections) == 1
    d = tr.detections[0]
    assert d.hits == 20 and d.max_count == 2 and d.max_score == pytest.approx(0.69)
    b = pt(5, north_m=50)                     # otro lugar
    tr.add_detection(5.0, b.lat, b.lon, 10.0, 1, 0.8)
    c = pt(60)                                # mismo lugar que a, mucho después
    tr.add_detection(60.0, c.lat, c.lon, 10.0, 1, 0.8)
    assert len(tr.detections) == 3


def test_exports_are_valid():
    tr = Trail()
    for i in range(5):
        tr.add(pt(i, north_m=3 * i))
    tr.add_detection(2.0, tr.points[2].lat, tr.points[2].lon, float('nan'), 3, 0.9)

    gj = json.loads(tr.geojson_str(current=tr.points[-1]))
    kinds = [f['geometry']['type'] for f in gj['features']]
    assert kinds == ['LineString', 'Point', 'Point', 'Point']
    assert len(gj['features'][0]['geometry']['coordinates']) == 5
    assert gj['features'][2]['properties']['personas_max'] == 3
    assert gj['features'][2]['properties']['alt_rel_m'] is None

    root = ET.fromstring(tr.to_kml())
    ns = {'k': 'http://www.opengis.net/kml/2.2'}
    assert len(root.findall('.//k:Placemark', ns)) == 3

    lines = tr.to_csv().strip().splitlines()
    assert len(lines) == 6 and lines[0].startswith('t,lat,lon')
    assert len(tr.detections_csv().strip().splitlines()) == 2


def test_empty_trail_exports():
    tr = Trail()
    assert tr.to_geojson()['features'] == []
    ET.fromstring(tr.to_kml())

