#!/usr/bin/env python3
"""Calibrar la ubicación de personas contra la verdad de terreno del SITL.

Toma un rosbag de un vuelo (detecciones + estado del dron) y las posiciones
reales de las personas (verdad.csv de verdad_sitl.py) y vuelve a ubicar CADA
detección con el mismo geolocate.locate() del nodo, probando correcciones:

  dt       desfase de tiempo: estado del dron en (stamp de la imagen + dt)
  pitch    corrección al pitch de montaje de la cámara (grados)
  altura   corrección a la altura sobre el suelo (m): terreno no plano,
           persona en una loma, lidar mal montado

Para cada combinación mide el error real (distancia a la persona más cercana)
y muestra la mejor contra la actual. Un error que se corrige con un parámetro
es sistemático; lo que queda es ruido.

Grabar (en otra terminal, mientras se vuela cerca de las personas a 15-30 m):
  ros2 bag record -o ~/bags/calib1 /yolo/detections /drone/attitude /drone/gps \\
      /drone/pose /airsim_node/PX4/distance/down
Después (con Unreal en Play, para la verdad):
  python3 scripts/verdad_sitl.py --personas "StaticMeshActor.*" --dron PX4
  python3 scripts/calibrar_geo.py --bag ~/bags/calib1 --verdad ~/yolo_flights/<vuelo>/verdad.csv
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src' / 'person_yolo'))
from person_yolo.geolocate import Camera, cos_tilt, locate  # noqa: E402
from person_yolo.history import History  # noqa: E402
from person_yolo.trail import haversine_m  # noqa: E402

TOPE_M = 30.0          # error real truncado: una falsa alarma no domina el promedio


def parse_args():
    ap = argparse.ArgumentParser(description='Calibrar geolocate contra la verdad del SITL.')
    ap.add_argument('--bag', required=True, type=Path)
    ap.add_argument('--verdad', required=True, type=Path, help='verdad.csv de verdad_sitl.py')
    ap.add_argument('--config', type=Path,
                    default=REPO / 'src' / 'person_yolo' / 'config' / 'sitl.yaml')
    ap.add_argument('--ancho', type=int, default=1920, help='ancho de la imagen de la cámara')
    ap.add_argument('--alto', type=int, default=1080)
    ap.add_argument('--todas', action='store_true',
                    help='usar también cajas sin confirmar por el tracker (más ruido)')
    return ap.parse_args()


def stamp_s(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def leer_bag(path, agl_topic):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    meta = yaml.safe_load((path / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=meta['storage_identifier']),
                rosbag2_py.ConverterOptions('', ''))
    tipos = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    falta = {'/yolo/detections', '/drone/attitude', '/drone/gps'} - set(tipos)
    if falta:
        sys.exit(f'Al bag le faltan {sorted(falta)}')
    att, gps, agl = History(1e9), History(1e9), History(1e9)
    dets = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in tipos:
            continue
        m = deserialize_message(data, tipos[topic])
        if topic == '/drone/attitude':
            q = m.quaternion
            att.add(stamp_s(m), (q.w, q.x, q.y, q.z))
        elif topic == '/drone/gps':
            if math.isfinite(m.latitude) and m.status.status >= 0:
                gps.add(stamp_s(m), (m.latitude, m.longitude))
        elif topic == agl_topic:
            if math.isfinite(m.range) and m.min_range <= m.range < m.max_range:
                agl.add(stamp_s(m), m.range)
        elif topic == '/yolo/detections':
            for d in m.detections:
                dets.append((stamp_s(m), d.id, d.bbox.center.position.x,
                             d.bbox.center.position.y))
    return dets, att, gps, agl


def config_geo(path):
    g = yaml.safe_load(path.read_text())['person_detector']['ros__parameters'].get(
        'geolocate', {})
    return {'hfov': g.get('camera_hfov_deg', 90.0), 'pitch': g.get('camera_pitch_deg', -30.0),
            'hperson': g.get('person_height_m', 0.9), 'inc': g.get('max_incidence_deg', 65.0),
            'slant': g.get('agl_is_slant', False), 'agl_topic': g.get('agl_topic', '/drone/agl')}


def main():
    a = parse_args()
    cfg = config_geo(a.config)
    verdad = [(r['actor'], float(r['lat']), float(r['lon']))
              for r in csv.DictReader(open(a.verdad.expanduser()))]
    dets, att, gps, agl = leer_bag(a.bag.expanduser(), cfg['agl_topic'])
    if not a.todas:
        dets = [d for d in dets if d[1]]           # solo tracks confirmados
    print(f'{len(dets)} detecciones, {len(att)} actitudes, {len(gps)} GPS, {len(agl)} lidar | '
          f'{len(verdad)} personas reales | config {a.config.name}: pitch {cfg["pitch"]}°, '
          f'hfov {cfg["hfov"]}°, incidencia máx {cfg["inc"]}°')
    if not dets:
        sys.exit('No hay detecciones confirmadas en el bag (¿se voló cerca de las personas?).')
    if len(agl) == 0:
        sys.exit(f'Sin lidar en {cfg["agl_topic"]}: no puedo ubicar (el nodo usaría la '
                 'altura sobre home).')
    calibrar(dets, att, gps, agl, verdad, cfg, a.ancho, a.alto)


def calibrar(dets, att, gps, agl, verdad, cfg, ancho, alto):
    """Busca (dt, pitch, altura) que minimizan el error real. Devuelve la mejor."""
    hfov, pitch0, hperson, inc, slant = (cfg[k] for k in ('hfov', 'pitch', 'hperson', 'inc',
                                                          'slant'))

    def evaluar(dt, dpitch, dalt, detalle=False):
        cam = Camera(hfov, pitch0 + dpitch)
        errs, dn, de = [], [], []
        for t, _, u, v in dets:
            s_att, s_gps = att.at(t + dt, 0.25), gps.at(t + dt, 0.5)
            s_agl = agl.at(t + dt, 0.5)
            if not (s_att and s_gps and s_agl):
                continue
            q = s_att[0]
            h = s_agl[0] * (max(0.0, cos_tilt(q)) if slant else 1.0) + dalt
            r = locate(u, v, ancho, alto, cam, q, h, s_gps[0][0], s_gps[0][1],
                       person_height_m=hperson, max_incidence_deg=inc)
            if r is None:
                continue
            d, (_, la, lo) = min((haversine_m(r['lat'], r['lon'], la, lo), (n, la, lo))
                                 for n, la, lo in verdad)
            errs.append(d)
            if detalle and d < TOPE_M:
                dn.append((r['lat'] - la) * 111_195)
                de.append((r['lon'] - lo) * 111_195 * math.cos(math.radians(la)))
        if not errs:
            return None
        errs.sort()
        res = {'n': len(errs), 'media_trunc': sum(min(e, TOPE_M) for e in errs) / len(errs),
               'mediana': errs[len(errs) // 2],
               'p90': errs[int(0.9 * (len(errs) - 1))],
               'en5m': sum(e <= 5 for e in errs) / len(errs)}
        if detalle and dn:
            res['sesgo'] = (sum(dn) / len(dn), sum(de) / len(de))
        return res

    def linea(nombre, dt, dp, dh, r):
        s = (f'  sesgo {r["sesgo"][0]:+.1f} m N, {r["sesgo"][1]:+.1f} m E'
             if 'sesgo' in r else '')
        print(f'{nombre:9s} dt {dt:+.2f} s, pitch {dp:+.0f}°, altura {dh:+.1f} m -> '
              f'mediana {r["mediana"]:5.1f} m, p90 {r["p90"]:5.1f} m, '
              f'{r["en5m"]:4.0%} a <5 m ({r["n"]} ubicadas){s}')

    base = evaluar(0.0, 0.0, 0.0, detalle=True)
    if base is None:
        print('Ninguna detección se pudo ubicar con estos datos.')
        return None
    linea('actual', 0.0, 0.0, 0.0, base)

    # Búsqueda en grilla: dt x pitch, y después altura con lo mejor.
    mejor = (base['media_trunc'], 0.0, 0.0, 0.0)
    for i in range(-15, 11):                       # dt de -1.5 a +1.0 s
        dt = i / 10
        for dp in range(-10, 11):
            r = evaluar(dt, dp, 0.0)
            if r and r['n'] >= 0.5 * base['n'] and r['media_trunc'] < mejor[0]:
                mejor = (r['media_trunc'], dt, dp, 0.0)
    _, dt, dp, _ = mejor
    for dh10 in range(-40, 41, 5):                 # altura de -4 a +4 m
        dh = dh10 / 10
        r = evaluar(dt, dp, dh)
        if r and r['n'] >= 0.5 * base['n'] and r['media_trunc'] < mejor[0]:
            mejor = (r['media_trunc'], dt, dp, dh)
    _, dt, dp, dh = mejor
    linea('mejor', dt, dp, dh, evaluar(dt, dp, dh, detalle=True))
    # Cada corrección sola, para ver cuál pesa.
    for nombre, args in (('solo dt', (dt, 0, 0)), ('solo pitch', (0, dp, 0)),
                         ('solo alt', (0, 0, dh))):
        r = evaluar(*args, detalle=True)
        if r:
            linea(nombre, *args, r)

    print('\nCómo leerlo: si "mejor" baja mucho la mediana, el error era sistemático.')
    print(f'  dt {dt:+.2f} s  -> geolocate.image_delay_sec: {-dt:.2f}  (positivo = la imagen '
          'es anterior a su stamp)')
    print(f'  pitch {dp:+.0f}° -> geolocate.camera_pitch_deg: {pitch0 + dp:.0f}  (revisar que '
          'coincida con el Pitch de yolo_cam en settings.json)')
    print(f'  altura {dh:+.1f} m -> terreno no plano o lidar: no hay parámetro directo; '
          'si es grande, avisar')
    return dt, dp, dh


if __name__ == '__main__':
    main()
