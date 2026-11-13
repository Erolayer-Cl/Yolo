import csv
import io
import json
import math
import xml.etree.ElementTree as ET

import pytest

from person_yolo.trail import PersonSpot, Trail, TrailPoint, haversine_m, spot_from_alert

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


def person(pid, located=True, north_m=0.0):
    p, d = pt(pid, north_m=north_m), pt(pid)
    return PersonSpot(float(pid), pid, p.lat, p.lon, located,
                      4.0 if located else float('nan'), 0.8, d.lat, d.lon)


def test_two_people_close_together_stay_two_points():
    tr = Trail()
    tr.add_person(person(1, north_m=10.0))
    tr.add_person(person(2, north_m=10.5))     # al lado: igual son dos personas
    assert len(tr.persons) == 2 and tr.dirty


def test_exports_are_valid():
    tr = Trail()
    for i in range(5):
        tr.add(pt(i, north_m=3 * i))
    tr.add_person(person(1, north_m=20.0))
    tr.add_person(person(2, located=False))

    gj = json.loads(tr.geojson_str())
    kinds = [f['geometry']['type'] for f in gj['features']]
    # recorrido, home, anillo de la persona ubicada, 2 personas
    assert kinds == ['LineString', 'Point', 'LineString', 'Point', 'Point']
    assert len(gj['features'][0]['geometry']['coordinates']) == 5
    p1, p2 = gj['features'][3]['properties'], gj['features'][4]['properties']
    assert p1['name'].startswith('persona #1: ') and p1['ubicada'] and p1['error_m'] == 4.0
    assert not p2['ubicada'] and p2['error_m'] is None
    assert p1['style']['color'] != p2['style']['color']

    root = ET.fromstring(tr.to_kml())
    ns = {'k': 'http://www.opengis.net/kml/2.2'}
    assert len(root.findall('.//k:Placemark', ns)) == 4      # ruta, home, 2 personas

    lines = tr.to_csv().strip().splitlines()
    assert len(lines) == 6 and lines[0].startswith('t,lat,lon')
    rows = tr.persons_csv().strip().splitlines()
    assert rows[0].startswith('id,t,lat,lon,ubicada,error_m') and len(rows) == 3
    assert rows[2].split(',')[4] == '0' and rows[2].split(',')[5] == ''


def test_empty_trail_exports():
    tr = Trail()
    assert tr.to_geojson()['features'] == []
    assert tr.rescue_waypoints() == []
    assert tr.grid_features(50.0) == []
    ET.fromstring(tr.to_kml())


def trail_with_people():
    tr = Trail()
    tr.add(pt(0))
    tr.add_person(person(1, north_m=30.0))
    tr.add_person(person(2, located=False))
    return tr


def test_tooltip_shows_rescue_coordinates():
    tr = trail_with_people()
    spot = tr.persons[0]
    feats = {f['properties'].get('name'): f for f in tr.to_geojson()['features']}
    name = f'persona #1: {spot.lat:.6f}, {spot.lon:.6f} ±4.0 m'
    meta = feats[name]['properties']['metadata']
    assert meta['lat'] == f'{spot.lat:.6f}' and meta['lon'] == f'{spot.lon:.6f}'
    assert meta['error (1σ)'] == '4.0 m'
    assert meta['desde home'] == '30 m N, 0 m E'
    assert meta['distancia a home'] == '30 m'


def test_unlocated_person_is_not_shown_as_rescue_coordinates():
    tr = trail_with_people()
    feats = {f['properties'].get('name'): f for f in tr.to_geojson()['features']}
    meta = feats['persona #2: APROXIMADA, posición del dron']['properties']['metadata']
    assert 'lat' not in meta and 'lon' not in meta and 'lat dron' in meta


def test_search_ring_is_closed_two_sigma_and_only_for_located():
    tr = trail_with_people()
    rings = [f for f in tr.to_geojson()['features']
             if f['properties'].get('name', '').startswith('zona de búsqueda')]
    assert len(rings) == 1 and 'persona #1' in rings[0]['properties']['name']
    coords = rings[0]['geometry']['coordinates']
    assert coords[0] == coords[-1]
    spot = tr.persons[0]
    for lon, lat in coords:
        assert haversine_m(spot.lat, spot.lon, lat, lon) == pytest.approx(8.0, abs=0.01)


def test_rescue_waypoints():
    tr = trail_with_people()
    located, unlocated = tr.rescue_waypoints()
    assert located['id'] == 1 and located['ubicada']
    assert located['lat'] == pytest.approx(tr.persons[0].lat, abs=1e-7)
    assert located['norte_m'] == pytest.approx(30.0, abs=0.1)
    assert located['este_m'] == pytest.approx(0.0, abs=0.1)
    assert located['distancia_home_m'] == pytest.approx(30.0, abs=0.1)
    assert unlocated['id'] == 2 and not unlocated['ubicada']
    assert unlocated['lat'] is None and unlocated['lon'] is None
    assert unlocated['error_m'] is None and unlocated['norte_m'] is None
    assert unlocated['drone_lat'] == pytest.approx(tr.persons[1].drone_lat, abs=1e-7)
    json.dumps(tr.rescue_waypoints())


def grid_names(gj):
    return [f['properties']['name'] for f in gj['features']
            if f['properties'].get('name', '').endswith(('N', 'S', 'E', 'O', '(home)'))]


def test_grid_only_in_live_message():
    tr = trail_with_people()
    assert grid_names(tr.to_geojson()) == []
    live = json.loads(tr.geojson_str(grid_step_m=50.0))
    names = grid_names(live)
    assert '0 m NS (home)' in names and '0 m EO (home)' in names
    assert '50 m N' in names and '50 m S' in names and '50 m E' in names and '50 m O' in names
    assert grid_names(json.loads(tr.geojson_str())) == []


def test_grid_covers_people_and_widens_step_when_too_many_lines():
    tr = Trail()
    tr.add(pt(0))
    tr.add_person(person(1, north_m=10_000.0))
    feats = tr.grid_features(50.0)
    names = [f['properties']['name'] for f in feats]
    assert len([n for n in names if n.endswith(('N', 'S', 'NS (home)'))]) <= 41
    lats = [c[1] for f in feats for c in f['geometry']['coordinates']]
    assert max(lats) > tr.persons[0].lat



def alert(pid=1, t=100.0, **kw):
    al = {'t': t, 'id': pid, 'score': 0.7, 'lat': LAT0, 'lon': LON0,
          'person_lat': None, 'person_lon': None, 'person_error_m': None}
    al.update(kw)
    return al


def test_better_fix_replaces_unlocated_person():
    tr = Trail()
    tr.add_person(spot_from_alert(alert()))
    assert not tr.persons[0].located
    tr.add_person(spot_from_alert(alert(person_lat=-33.5, person_lon=-70.5, person_error_m=3.0)))
    assert len(tr.persons) == 1
    assert tr.persons[0].located and tr.persons[0].lat == -33.5


def test_unlocated_never_overwrites_located():
    tr = Trail()
    tr.add_person(spot_from_alert(alert(person_lat=-33.5, person_lon=-70.5, person_error_m=3.0)))
    tr.add_person(spot_from_alert(alert()))
    assert tr.persons[0].located and tr.persons[0].lat == -33.5


def test_repeated_id_with_other_time_is_another_person():
    # El detector se reinició: vuelve a dar el #1, pero no es la misma persona.
    tr = Trail()
    tr.add_person(spot_from_alert(alert(t=100.0)))
    tr.add_person(spot_from_alert(alert(t=500.0)))
    assert len(tr.persons) == 2


@pytest.mark.parametrize('bad', [
    [], 'texto', None, {}, {'lat': None, 'lon': None},
    alert(lat=float('nan')), alert(lon=float('inf')), alert(lat=95.0),
    alert(t=float('nan')), alert(pid='x'), alert(lat='x'),
])
def test_bad_alert_is_rejected(bad):
    assert spot_from_alert(bad) is None


def test_bad_person_position_falls_back_to_unlocated():
    spot = spot_from_alert(alert(person_lat=float('nan'), person_lon=-70.5, person_error_m=2.0))
    assert not spot.located and spot.lat == LAT0 and math.isnan(spot.error_m)


def test_manual_marks_are_separate_and_deletable():
    tr = Trail()
    tr.add(pt(0.0))
    tr.add_person(person(1))
    m1 = tr.add_mark(10.0, LAT0, LON0)
    m2 = tr.add_mark(11.0, LAT0 + 1e-4, LON0)
    assert (m1.label, m2.label) == ('M1', 'M2') and m1.id < 0 and m2.id < 0
    wps = {w['etiqueta']: w for w in tr.rescue_waypoints()}
    assert wps['M1']['manual'] and wps['M1']['lat'] == round(LAT0, 8)
    assert not wps['1']['manual']
    csv_text = tr.persons_csv()
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert [r['id'] for r in rows] == ['1', '-1', '-2']        # id numérico siempre
    assert [r['etiqueta'] for r in rows] == ['1', 'M1', 'M2']
    assert [r['manual'] for r in rows] == ['0', '1', '1']
    assert not tr.remove_mark(1)                   # las del detector no se borran a mano
    assert tr.remove_mark(m1.id) and [p.label for p in tr.persons] == ['1', 'M2']
    assert tr.add_mark(12.0, LAT0, LON0).label == 'M3'   # los ids no se reutilizan
    json.dumps(tr.to_geojson())
    ET.fromstring(tr.to_kml())


def test_merged_person_is_removed_but_not_manual_marks():
    tr = Trail()
    tr.add(pt(0.0))
    a, b = person(1), person(2)
    tr.add_person(a)
    tr.add_person(b)
    tr.add_mark(5.0, LAT0, LON0)
    assert tr.remove_person(2, b.t)
    assert not tr.remove_person(2, b.t)            # ya no está
    assert [p.label for p in tr.persons] == ['1', 'M1']


def test_detected_point_deleted_from_web_stays_deleted():
    tr = Trail()
    tr.add(pt(0.0))
    a, b = person(1), person(2)
    tr.add_person(a)
    tr.add_person(b)
    m = tr.add_mark(5.0, LAT0, LON0)
    t_web = round(b.t, 2)                          # la web manda t redondeado
    assert not tr.remove_detected(2, t_web + 1.0)  # otro t: otra persona
    assert not tr.remove_detected(m.id, m.t)       # las marcas van por remove_mark
    assert tr.remove_detected(2, t_web)
    assert [p.label for p in tr.persons] == ['1', 'M1']
    tr.add_person(person(2))                       # un fix posterior no la recrea
    assert [p.label for p in tr.persons] == ['1', 'M1']
    assert not tr.remove_detected(2, t_web)        # ya no está


def test_late_fix_does_not_recreate_merged_person():
    tr = Trail()
    tr.add(pt(0.0))
    b = person(2)
    tr.add_person(b)
    assert tr.remove_person(2, b.t)
    tr.add_person(person(2))                     # fix de #2 que ya estaba en la cola
    assert [p.label for p in tr.persons] == []
    other = person(2)
    other.t += 50.0                              # mismo id tras reiniciar el detector: otra
    tr.add_person(other)
    assert [p.label for p in tr.persons] == ['2']
