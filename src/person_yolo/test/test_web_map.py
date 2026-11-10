import json
import urllib.error
import urllib.request

import pytest

from person_yolo.trail import PersonSpot, Trail, TrailPoint
from person_yolo.web_map import MapServer

LAT0, LON0 = -33.4670, -70.6570


@pytest.fixture
def server():
    s = MapServer(0, '127.0.0.1')               # puerto libre cualquiera
    yield s
    s.stop()


def get(server, path, headers=None):
    req = urllib.request.Request(f'http://127.0.0.1:{server.port}{path}', headers=headers or {})
    return urllib.request.urlopen(req, timeout=5)


def test_page_is_served(server):
    r = get(server, '/')
    body = r.read().decode()
    assert r.status == 200 and 'text/html' in r.headers['Content-Type']
    assert '<canvas id="map"' in body and '/api/state' in body


def test_api_and_csv(server):
    server.set('/api/state', b'{"persons": []}')
    assert json.loads(get(server, '/api/state').read()) == {'persons': []}
    server.set('/personas.csv', b'id,t\n')
    r = get(server, '/personas.csv')
    assert r.read() == b'id,t\n' and 'attachment' in r.headers['Content-Disposition']


def test_unchanged_data_gives_304(server):
    server.set('/api/live', b'{"lat": 1}')
    etag = get(server, '/api/live').headers['ETag']
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(server, '/api/live', {'If-None-Match': etag})
    assert exc.value.code == 304
    server.set('/api/live', b'{"lat": 2}')
    assert json.loads(get(server, '/api/live', {'If-None-Match': etag}).read()) == {'lat': 2}


def test_unknown_path_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(server, '/../../etc/passwd')
    assert exc.value.code == 404


def test_web_state_in_metres_from_home():
    tr = Trail()
    tr.add(TrailPoint(0.0, LAT0, LON0, 520.0, 0.0, 0.0))
    north = LAT0 + 30.0 / 111_195.0
    tr.add(TrailPoint(1.0, north, LON0, 520.0, 10.0, 0.0))
    tr.add_person(PersonSpot(2.0, 1, north, LON0, True, 2.0, 0.8, LAT0, LON0))
    # Sin ubicar: su lat/lon es la del dron (así la arma spot_from_alert).
    tr.add_person(PersonSpot(3.0, 2, LAT0, LON0, False, float('nan'), 0.5, LAT0, LON0))
    s = json.loads(json.dumps(tr.web_state()))     # tiene que ser JSON válido (sin NaN)
    assert s['home'] == {'lat': LAT0, 'lon': LON0}
    assert s['trail'][1] == pytest.approx([30.0, 0.0], abs=0.05)
    p1, p2 = s['persons']
    assert p1['ubicada'] and p1['pos'] == pytest.approx([30.0, 0.0], abs=0.05)
    assert not p2['ubicada'] and p2['lat'] is None and p2['pos'] == [0.0, 0.0]


def test_web_state_without_home():
    s = Trail().web_state()
    assert s['home'] is None and s['trail'] == [] and s['persons'] == []


def post(server, path, body=None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{server.port}{path}', data=data,
                                 headers={'Content-Type': 'application/json'}, method='POST')
    return urllib.request.urlopen(req, timeout=5)


def test_post_goes_to_callback():
    calls = []

    def on_post(path, data):
        calls.append((path, data))
        return 200, {'ok': True, 'lat': 1.5}

    s = MapServer(0, '127.0.0.1', on_post=on_post)
    try:
        r = post(s, '/api/mark/delete', {'id': -2})
        assert json.loads(r.read()) == {'ok': True, 'lat': 1.5}
        assert calls == [('/api/mark/delete', {'id': -2})]
    finally:
        s.stop()


def test_post_errors_are_reported_not_crashes():
    def on_post(path, data):
        if data.get('boom'):
            raise RuntimeError('falló')
        return 409, {'error': 'sin GPS del dron'}

    s = MapServer(0, '127.0.0.1', on_post=on_post)
    try:
        for body, code in (({}, 409), ({'boom': 1}, 500)):
            with pytest.raises(urllib.error.HTTPError) as exc:
                post(s, '/api/mark', body)
            assert exc.value.code == code and 'error' in json.loads(exc.value.read())
        assert get(s, '/').status == 200            # el servidor sigue vivo
    finally:
        s.stop()


def test_post_disabled_without_callback(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(server, '/api/mark')
    assert exc.value.code == 404


def test_page_has_mark_button(server):
    body = get(server, '/').read().decode()
    assert 'id="bMark"' in body and '/api/mark' in body
