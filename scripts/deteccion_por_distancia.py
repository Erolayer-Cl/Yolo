#!/usr/bin/env python3
"""Tasa de detección por distancia, para cada modelo, desde rosbags del SITL.

Cada rosbag es UNA corrida: el dron en hover frente a UNA persona a una distancia
horizontal conocida (o la escena sin personas). Cada frame de /camera pasa por
detect_persons() (la misma función del nodo, vía training/e4_comparar_sitl.py)
con los parámetros de config/sitl.yaml, para cada modelo. Por corrida y modelo:

  frames                 frames leídos del bag
  frames_con_deteccion   frames con al menos una caja
  tasa                   frames_con_deteccion / frames
  cajas_por_frame        cajas en total / frames

Escena vacía: toda caja es una falsa alarma, así que cajas_por_frame = falsas
alarmas por frame. Limitación: en una corrida CON persona, una falsa alarma en
un frame también cuenta como detección (no hay etiquetas por caja). Se acota con
la tasa de la escena vacía y mirando los frames a mano.

Distancia máxima de detección (definición del informe): la mayor distancia con
tasa >= 50 %, por modelo y altura.

Los datos de la corrida salen del nombre del bag:
  dist_h15_d10_r1            altura 15 m, distancia 10 m, de pie, repetición 1
  dist_h15_d50_tendida_r2    con pose
  dist_h15_vacia_r1          escena sin personas
o de --altura/--distancia/--pose/--rep (solo con un bag).

Uso (con ROS para leer el bag):
  source /opt/ros/jazzy/setup.bash
  python3 scripts/deteccion_por_distancia.py ~/bags/dist_h15_d10_r1 ~/bags/dist_h15_d25_r1 ...
"""
import argparse
import csv
import re
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
# training/ para reutilizar e4 (lectura del bag, modelo) y el paquete para la inferencia.
sys.path[:0] = [str(REPO / 'training'), str(REPO / 'src' / 'person_yolo')]

COLUMNAS = ['altura_m', 'distancia_m', 'pose', 'repeticion', 'modelo', 'frames',
            'frames_con_deteccion', 'tasa', 'cajas_por_frame']
MODELOS = ['coco=yolo11n.pt', 'n_final=' + str(REPO / 'models' / 'visdrone_n.pt')]
UMBRAL_TASA = 0.5           # definición de "detecta" a una distancia (informe)
# dist_h<altura>_(d<distancia>|vacia)[_<pose>]_r<repetición>
NOMBRE_BAG = re.compile(r'dist_h(?P<altura>\d+(?:\.\d+)?)_'
                        r'(?:d(?P<distancia>\d+(?:\.\d+)?)|(?P<vacia>vacia))'
                        r'(?:_(?P<pose>[a-z]+))?_r(?P<rep>\d+)$')


def datos_de_la_corrida(nombre):
    """Altura, distancia (None = escena vacía), pose y repetición desde el nombre
    del bag. None si el nombre no sigue el formato."""
    m = NOMBRE_BAG.match(nombre)
    if m is None:
        return None
    vacia = m['vacia'] is not None
    return {'altura_m': float(m['altura']),
            'distancia_m': None if vacia else float(m['distancia']),
            'pose': 'vacia' if vacia else (m['pose'] or 'de_pie'),
            'repeticion': int(m['rep'])}


def contar(cajas_por_frame):
    """Lista con la cantidad de cajas de cada frame -> los cuatro números de la fila."""
    frames = len(cajas_por_frame)
    con_deteccion = sum(1 for n in cajas_por_frame if n > 0)
    return {'frames': frames,
            'frames_con_deteccion': con_deteccion,
            'tasa': con_deteccion / frames if frames else 0.0,
            'cajas_por_frame': sum(cajas_por_frame) / frames if frames else 0.0}


def distancia_maxima(filas, umbral=UMBRAL_TASA):
    """{(modelo, altura): mayor distancia con tasa >= umbral, o None si ninguna}.
    Las corridas sin persona no cuentan."""
    out = {}
    for f in filas:
        if f['distancia_m'] is None:
            continue
        clave = (f['modelo'], f['altura_m'])
        out.setdefault(clave, None)
        if f['tasa'] >= umbral and (out[clave] is None or f['distancia_m'] > out[clave]):
            out[clave] = f['distancia_m']
    return out


def parse_args():
    ap = argparse.ArgumentParser(description='Tasa de detección por distancia (rosbags del SITL).')
    ap.add_argument('bags', nargs='+', type=Path, help='carpetas de rosbag2, una por corrida')
    ap.add_argument('--model', action='append', metavar='NOMBRE=PESOS',
                    help='repetir por modelo (por defecto coco y n_final)')
    ap.add_argument('--config', type=Path,
                    default=REPO / 'src' / 'person_yolo' / 'config' / 'sitl.yaml')
    ap.add_argument('--topic', default='/camera')
    ap.add_argument('--salto', type=int, default=1, help='usar 1 de cada N frames')
    ap.add_argument('--max-frames', type=int, default=0, help='0 = todos')
    ap.add_argument('--device', default='', help="'' = automático, 'cpu', '0'...")
    ap.add_argument('--altura', type=float, help='solo con un bag: altura de vuelo (m)')
    ap.add_argument('--distancia', type=float, help='solo con un bag: distancia horizontal (m)')
    ap.add_argument('--pose', help='solo con un bag: de_pie, sentada, tendida, vacia')
    ap.add_argument('--rep', type=int, help='solo con un bag: repetición')
    ap.add_argument('--csv', type=Path,
                    default=REPO / 'docs' / 'resultados' / 'sitl_distancia' /
                    'deteccion_por_distancia.csv')
    return ap.parse_args()


def datos_de(bag, a):
    """Datos de la corrida: del nombre del bag, pisados por los argumentos si hay un solo bag."""
    datos = datos_de_la_corrida(bag.name) or {}
    if len(a.bags) == 1:
        for campo, valor in (('altura_m', a.altura), ('distancia_m', a.distancia),
                             ('pose', a.pose), ('repeticion', a.rep)):
            if valor is not None:
                datos[campo] = valor
    if set(datos) != {'altura_m', 'distancia_m', 'pose', 'repeticion'}:
        raise SystemExit(f'{bag.name}: el nombre no dice altura/distancia/repetición '
                         '(ej. dist_h15_d10_r1); con un solo bag, usar --altura --distancia --rep')
    return datos


def main():
    a = parse_args()
    # Importaciones pesadas acá: el test de contar() no necesita GPU, ROS ni OpenCV.
    from e3_evaluar import import_inference
    from e4_comparar_sitl import Corredor, frames, leer_config
    from person_yolo.tracker import PersonCounter
    InferenceParams, detect_persons, _ = import_inference(REPO)

    cfg = leer_config(a.config.expanduser(), InferenceParams)
    print(f'Parámetros ({a.config.name}): {cfg["params"]}; tiles cada {cfg["every"]} frames')
    opciones = SimpleNamespace(device=a.device, tiles_siempre=False)   # lo que lee Corredor
    corredores = []
    for spec in a.model or MODELOS:
        nombre, _, pesos = spec.partition('=')
        print(f'Cargando {nombre} ({pesos})')
        corredores.append(Corredor(nombre, pesos, cfg, opciones, PersonCounter))

    filas = []
    for bag in (b.expanduser().resolve() for b in a.bags):
        datos = datos_de(bag, a)
        fuente = SimpleNamespace(src=bag, topic=a.topic, fps=10.0, salto=a.salto,
                                 max_frames=a.max_frames)             # lo que lee frames()
        cajas = {c.nombre: [] for c in corredores}
        for k, (_, frame) in enumerate(frames(fuente)):
            use_tiles = k % cfg['every'] == 0                          # como en vuelo
            for c in corredores:
                r = detect_persons(frame, c.predict_fn, c.params, use_tiles=use_tiles)
                cajas[c.nombre].append(len(r.boxes))
        for c in corredores:
            fila = {**datos, 'modelo': c.nombre, **contar(cajas[c.nombre])}
            filas.append(fila)
            print(f'{bag.name}  {c.nombre:8s} frames {fila["frames"]:4d}  con detección '
                  f'{fila["frames_con_deteccion"]:4d}  tasa {fila["tasa"]:.0%}  '
                  f'cajas/frame {fila["cajas_por_frame"]:.2f}')

    a.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        for fila in filas:
            w.writerow({**fila, 'distancia_m': '' if fila['distancia_m'] is None
                        else fila['distancia_m'],
                        'tasa': f'{fila["tasa"]:.3f}',
                        'cajas_por_frame': f'{fila["cajas_por_frame"]:.3f}'})
    print(f'\nCSV: {a.csv}')
    for (modelo, altura), d in sorted(distancia_maxima(filas).items()):
        print(f'Distancia máxima con tasa >= {UMBRAL_TASA:.0%}: {modelo} a {altura:g} m de '
              f'altura -> {"ninguna" if d is None else f"{d:g} m"}')
    for fila in filas:
        if fila['distancia_m'] is None:
            print(f'Escena vacía (h{fila["altura_m"]:g} r{fila["repeticion"]}): {fila["modelo"]} '
                  f'{fila["cajas_por_frame"]:.2f} falsas alarmas por frame')


if __name__ == '__main__':
    main()
