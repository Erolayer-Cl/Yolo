#!/usr/bin/env python3
"""Verdad de terreno en SITL: dónde están DE VERDAD las personas en Unreal, y
cuánto se equivocan los puntos del detector.

AirSim conoce la pose de cada actor de la escena. Este script la convierte a
lat/lon con el GPS simulado del dron (el mismo que recibe PX4, y por lo tanto
/drone/gps) y la compara con personas.csv de un vuelo:

  - error REAL de cada punto (distancia a la persona más cercana) contra el
    error_m que dice el detector
  - duplicados: varios puntos que son la misma persona
  - falsas alarmas: puntos sin ninguna persona cerca
  - personas que nunca se marcaron

Uso (con Unreal en Play; no hace falta ROS):
  # 1. ver cómo se llaman los actores, para elegir el patrón de las personas
  python3 scripts/verdad_sitl.py --listar
  # 2. comparar con el último vuelo (o --vuelo ~/yolo_flights/<carpeta>)
  python3 scripts/verdad_sitl.py --personas 'BP_Persona.*'
Escribe verdad.csv y verdad.geojson en la carpeta del vuelo (el geojson se
abre en geojson.io junto a trail.geojson).

Supone personas quietas: la posición es la de cuando corre el script.
"""
import argparse
import collections
import csv
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AIRSIM_PY = Path(os.environ.get(
    'AIRSIM_PY', REPO.parent.parent / 'dependencias' / 'Cosys-AirSim' / 'PythonClient'))
sys.path[:0] = [str(AIRSIM_PY), str(REPO / 'src' / 'person_yolo')]

from person_yolo.geolocate import offset_to_latlon  # noqa: E402
from person_yolo.trail import haversine_m  # noqa: E402

FALSA_ALARMA_M = 15.0      # un punto a más de esto de toda persona = falsa alarma


def parse_args():
    ap = argparse.ArgumentParser(description='Verdad de terreno de personas en SITL.')
    ap.add_argument('--listar', action='store_true', help='listar los actores de la escena')
    ap.add_argument('--personas', default=r'(?i).*(person|persona|human|humano|character|'
                    r'metahuman|mannequin|maniqui|npc|civil).*',
                    help='regex de los nombres de actor de las personas')
    ap.add_argument('--vuelo', type=Path, help='carpeta del vuelo (por defecto la última '
                    'de ~/yolo_flights)')
    ap.add_argument('--sin-airsim', action='store_true',
                    help='no conectarse a Unreal: rehacer el informe con el verdad.csv ya '
                    'guardado en la carpeta del vuelo (reproducible sin el simulador)')
    ap.add_argument('--vehiculo', default='PX4')
    ap.add_argument('--host', default='localhost')
    return ap.parse_args()


def conectar(a):
    import cosysairsim as airsim
    c = airsim.MultirotorClient(ip=a.host, port=41451, timeout_value=30)
    c.confirmConnection()
    return c


def listar(c):
    objs = c.simListSceneObjects('.*')
    grupos = collections.Counter(re.sub(r'[_\d]+$', '', o) for o in objs)
    print(f'{len(objs)} actores. Prefijos (cantidad, nombre sin el número final):')
    for nombre, n in sorted(grupos.items(), key=lambda kv: -kv[1]):
        print(f'  {n:5d}  {nombre}')


def posiciones(c, a):
    """[(actor, lat, lon)] de las personas, con el GPS simulado del dron."""
    objs = c.simListSceneObjects('.*')
    personas = [o for o in objs if re.fullmatch(a.personas, o) and o != a.vehiculo]
    if not personas:
        sys.exit(f'Ningún actor coincide con {a.personas!r} (tiene que calzar el nombre '
                 'completo): --listar y --personas')
    # Pose del dron y su GPS en el mismo momento: las personas se ubican
    # relativas al dron, así no importa dónde está el origen del mundo.
    # La pose es la del VEHÍCULO, no la de un actor buscado por nombre: el actor
    # SimModeWorldMultiRotor_0 también calza con "multirotor" pero no se mueve,
    # y con el dron lejos de él corría todas las personas (61 m el 2026-11-11).
    pd = c.simGetVehiclePose(vehicle_name=a.vehiculo).position
    if any(math.isnan(v) for v in (pd.x_val, pd.y_val)):
        sys.exit(f'AirSim no da la pose del vehículo {a.vehiculo!r}: ¿--vehiculo?')
    g = c.getGpsData(vehicle_name=a.vehiculo).gnss.geo_point
    out = []
    for p in personas:
        pp = c.simGetObjectPose(p).position
        if any(math.isnan(v) for v in (pp.x_val, pp.y_val)):
            continue
        lat, lon = offset_to_latlon(g.latitude, g.longitude,
                                    pp.x_val - pd.x_val, pp.y_val - pd.y_val)
        out.append((p, lat, lon))
    print(f'Dron: vehículo {a.vehiculo}, GPS {g.latitude:.6f}, {g.longitude:.6f}')
    return out


def ultimo_vuelo():
    vuelos = sorted((Path.home() / 'yolo_flights').glob('*/personas.csv'))
    return vuelos[-1].parent if vuelos else None


def _asignar_juntos(grupo, verdad):
    """Puntos marcados en el MISMO instante = personas distintas (el detector las vio
    a la vez en el mismo cuadro): se asignan uno a uno a personas reales distintas,
    con la menor distancia total. Si hay más puntos que personas, los que sobran van
    a la más cercana (son falsas alarmas o duplicados de verdad)."""
    k, n = len(grupo), len(verdad)
    dist = [[haversine_m(p['lat'], p['lon'], la, lo) for _, la, lo in verdad] for p in grupo]
    if k > 7:                          # no pasa en la práctica; evita la explosión
        return [min(range(n), key=lambda j: dist[i][j]) for i in range(k)]
    sin = 1e6                          # costo de dejar un punto sin persona propia
    mejor, asig = None, None
    for perm in itertools.permutations(list(range(n)) + [None] * k, k):
        costo = sum(sin if j is None else dist[i][j] for i, j in enumerate(perm))
        if mejor is None or costo < mejor:
            mejor, asig = costo, perm
    return [min(range(n), key=lambda j: dist[i][j]) if j is None else j
            for i, j in enumerate(asig)]


def comparar(verdad, puntos):
    """Para cada punto, la persona real que le corresponde y la distancia.

    Puntos del mismo instante: asignación uno a uno (ver _asignar_juntos). El resto:
    la persona más cercana, así una persona marcada dos veces en momentos distintos
    sigue contando como duplicado."""
    grupos = collections.defaultdict(list)
    for i, pt in enumerate(puntos):
        grupos[pt['t'] if pt['t'] is not None and not pt['manual'] else ('solo', i)].append(i)
    filas = [None] * len(puntos)
    for idx in grupos.values():
        grupo = [puntos[i] for i in idx]
        for i, j in zip(idx, _asignar_juntos(grupo, verdad)):
            nombre, lat, lon = verdad[j]
            filas[i] = {**puntos[i], 'real': nombre,
                        'err_real': haversine_m(puntos[i]['lat'], puntos[i]['lon'], lat, lon)}
    return filas


def leer_puntos(vuelo):
    with open(vuelo / 'personas.csv') as f:
        out = []
        for r in csv.DictReader(f):
            ubicada = r['ubicada'] in ('1', 'True', 'true')
            lat, lon = (r['lat'], r['lon']) if ubicada else (r['drone_lat'], r['drone_lon'])
            if not lat or not lon:
                continue
            manual = r.get('manual') in ('1', 'True', 'true')
            out.append({'id': r['id'], 'lat': float(lat), 'lon': float(lon),
                        't': r.get('t') or None,
                        'ubicada': ubicada, 'manual': manual,
                        'error_m': float(r['error_m']) if ubicada and r['error_m'] else None})
        return out


def informe(verdad, filas):
    print(f'\nPersonas reales: {len(verdad)} | puntos marcados: {len(filas)}\n')
    print(f'{"#":>3} {"estado":7} {"persona real asignada":28} {"error real":>10} '
          f'{"dice":>8}  ')
    for r in filas:
        dice = f'±{r["error_m"]:.0f} m' if r['error_m'] is not None else 'dron'
        lejos = r['err_real'] > FALSA_ALARMA_M
        if r['manual']:                 # a mano, en la posición del dron
            nota = 'marca manual'
        elif not r['ubicada']:          # el punto es el dron: no se espera que caiga encima
            nota = 'aprox: lejos de toda persona' if lejos else 'aprox: cerca de una persona'
        elif lejos:
            nota = 'FALSA ALARMA o muy mal ubicada'
        elif r['error_m'] is None:
            nota = 'ubicada, sin error informado'
        else:
            nota = 'ok' if r['err_real'] <= 2 * r['error_m'] else 'error MAYOR al que dice'
        estado = 'manual' if r['manual'] else 'GPS' if r['ubicada'] else 'aprox'
        print(f'{r["id"]:>3} {estado:7} {r["real"][:28]:28} '
              f'{r["err_real"]:8.1f} m {dice:>8}  {nota}')

    manuales = [r for r in filas if r['manual']]
    filas = [r for r in filas if not r['manual']]      # el resumen es del detector
    if manuales:
        print(f'\nMarcas manuales: {len(manuales)}, a ' + ', '.join(
            f'{r["err_real"]:.1f} m de {r["real"]}' for r in manuales))
    reales = [r for r in filas if r['err_real'] <= FALSA_ALARMA_M]
    aprox_lejos = sum(1 for r in filas if not r['ubicada'] and r['err_real'] > FALSA_ALARMA_M)
    por_persona = collections.Counter(r['real'] for r in reales)
    ubic = sorted(r['err_real'] for r in reales if r['ubicada'])
    print('\nResumen:')
    print(f'  personas encontradas: {len(por_persona)}/{len(verdad)}'
          + (f' (faltan: {", ".join(sorted(set(n for n, *_ in verdad) - set(por_persona)))})'
             if len(por_persona) < len(verdad) else ''))
    print(f'  puntos de más (duplicados): {sum(n - 1 for n in por_persona.values())}  '
          + ', '.join(f'{k}: {v}' for k, v in por_persona.items() if v > 1))
    print(f'  ubicados a > {FALSA_ALARMA_M:.0f} m de toda persona (falsa alarma o mal '
          f'ubicados): {len(filas) - len(reales) - aprox_lejos}')
    print(f'  aproximados lejos de toda persona (puntos sobre el dron, sin valor de '
          f'rescate): {aprox_lejos}')
    if ubic:
        print(f'  error real de los ubicados: mediana {ubic[len(ubic) // 2]:.1f} m, '
              f'máx {ubic[-1]:.1f} m')
    # Sesgo: si todos se corren para el mismo lado es un error sistemático
    # (pitch de la cámara, retraso), no ruido.
    if len(ubic) >= 3:
        dn = [(r['lat'] - next(la for n, la, _ in verdad if n == r['real'])) * 111_195
              for r in reales if r['ubicada']]
        de = [(r['lon'] - next(lo for n, _, lo in verdad if n == r['real'])) * 111_195
              * math.cos(math.radians(r['lat'])) for r in reales if r['ubicada']]
        print(f'  corrimiento medio de los ubicados: {sum(dn) / len(dn):+.1f} m norte, '
              f'{sum(de) / len(de):+.1f} m este (si es grande frente a la mediana: sesgo)')


def guardar(vuelo, verdad, filas):
    with open(vuelo / 'verdad.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['actor', 'lat', 'lon'])
        w.writerows(verdad)
    gj = {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
         'properties': {'name': n, 'marker-color': '#2e7d32', 'tipo': 'real'}}
        for n, lat, lon in verdad] + [
        {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [r['lon'], r['lat']]},
         'properties': {'name': f'#{r["id"]}', 'error_real_m': round(r['err_real'], 1),
                        'marker-color': '#d32f2f' if r['ubicada'] else '#ef6c00',
                        'tipo': 'detectada'}} for r in filas]}
    (vuelo / 'verdad.geojson').write_text(json.dumps(gj, indent=1))
    print(f'\nGuardado: {vuelo}/verdad.csv y verdad.geojson (verde = real, rojo/naranja = '
          'detectada)')


def leer_verdad(vuelo):
    with open(vuelo / 'verdad.csv') as f:
        return [(r['actor'], float(r['lat']), float(r['lon'])) for r in csv.DictReader(f)]


def main():
    a = parse_args()
    if a.sin_airsim:
        vuelo = a.vuelo.expanduser() if a.vuelo else ultimo_vuelo()
        if vuelo is None or not (vuelo / 'verdad.csv').exists():
            sys.exit('No hay verdad.csv en el vuelo: correr antes con Unreal en Play.')
        print(f'Vuelo: {vuelo} (verdad.csv guardado, sin AirSim)')
        verdad = leer_verdad(vuelo)
        informe(verdad, comparar(verdad, leer_puntos(vuelo)))
        return
    c = conectar(a)
    if a.listar:
        listar(c)
        return
    verdad = posiciones(c, a)
    print(f'{len(verdad)} personas: ' + ', '.join(f'{n} ({la:.6f}, {lo:.6f})'
                                                  for n, la, lo in verdad))
    vuelo = a.vuelo.expanduser() if a.vuelo else ultimo_vuelo()
    if vuelo is None or not (vuelo / 'personas.csv').exists():
        print('No hay personas.csv para comparar (¿ya volaste?).')
        return
    print(f'Vuelo: {vuelo}')
    filas = comparar(verdad, leer_puntos(vuelo))
    informe(verdad, filas)
    guardar(vuelo, verdad, filas)


if __name__ == '__main__':
    main()
