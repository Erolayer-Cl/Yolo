#!/usr/bin/env python3
"""Etapa 4: comparar pesos sobre los MISMOS frames del SITL (o de cualquier video), sin etiquetas.

e3_evaluar.py mide contra VisDrone (fotos reales con etiquetas). Esto responde
otra pregunta: en Unreal, que es lo que vuela hoy, ¿qué ve cada modelo? Cada
frame pasa por detect_persons() (la función del nodo) con los parámetros del
YAML, los tiles cada `every_n_frames` como en vuelo, y por el mismo tracker
(PersonCounter) para contar personas distintas.

Fuente (--src):
  - un rosbag2 (carpeta con metadata.yaml) con /camera:
        ros2 bag record /camera -o ~/bags/vuelo1      # durante un vuelo en SITL
  - un video (.mp4, .avi, .mkv...)
  - una carpeta de imágenes (.jpg/.png, en orden alfabético)

Salida en runs/comparar/<nombre de la fuente>/:
  - comparacion.mp4   mosaico, un panel por modelo, con conteo, total y ms
  - resumen.md/json   por modelo: personas por frame, personas distintas,
                      cajas que solo ve ese modelo, score, ms por frame
  - discrepancias/    los frames donde más difieren los modelos (para mirarlos)

Sin etiquetas no hay precisión ni recall: más cajas no es mejor si son falsas
alarmas. Mirar discrepancias/ y el video para decidir.

Uso (desde training/):
  python e4_comparar_sitl.py --src ~/bags/vuelo1
  python e4_comparar_sitl.py --src vuelo.mp4 --model n=../models/visdrone_n.pt --model s=../models/visdrone_s.pt
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from e2_entrenar import resolver_pesos
from e3_evaluar import fp16_kwargs, import_inference, iou_matrix, load_params, person_classes

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parent
MODELOS_POR_DEFECTO = ['coco=yolo11n.pt',
                       'n=' + str(REPO / 'models' / 'visdrone_n.pt'),
                       's=' + str(REPO / 'models' / 'visdrone_s.pt')]
IMAGENES = {'.jpg', '.jpeg', '.png', '.bmp'}


def parse_args():
    ap = argparse.ArgumentParser(description='Comparar modelos sobre los mismos frames.')
    ap.add_argument('--src', required=True, type=Path, help='rosbag2, video o carpeta de imágenes')
    ap.add_argument('--model', action='append', metavar='NOMBRE=PESOS',
                    help='repetir por modelo (por defecto coco, n y s)')
    ap.add_argument('--config', type=Path,
                    default=REPO / 'src' / 'person_yolo' / 'config' / 'sitl.yaml')
    ap.add_argument('--topic', default='/camera', help='tópico de imagen en el rosbag')
    ap.add_argument('--fps', type=float, default=10.0,
                    help='fps de una carpeta de imágenes (para el tracker)')
    ap.add_argument('--salto', type=int, default=1, help='usar 1 de cada N frames')
    ap.add_argument('--max-frames', type=int, default=0, help='0 = todos')
    ap.add_argument('--tiles-siempre', action='store_true',
                    help='tiles en todos los frames (mejor caso), no cada every_n_frames')
    ap.add_argument('--ancho-panel', type=int, default=960, help='ancho de cada panel del video')
    ap.add_argument('--discrepancias', type=int, default=8, help='frames a guardar')
    ap.add_argument('--device', default='', help="'' = automático, 'cpu', '0'...")
    ap.add_argument('--out', type=Path, default=AQUI / 'runs' / 'comparar')
    return ap.parse_args()


# --------------------------------------------------------------------- fuentes
def frames(a):
    """Genera (t en segundos, frame BGR) según el tipo de fuente."""
    src = a.src.expanduser()
    if (src / 'metadata.yaml').exists():
        gen = _frames_bag(src, a.topic)
    elif src.is_dir():
        gen = _frames_carpeta(src, a.fps)
    else:
        gen = _frames_video(src)
    n = 0
    for k, (t, img) in enumerate(gen):
        if k % a.salto:
            continue
        yield t, img
        n += 1
        if a.max_frames and n >= a.max_frames:
            return


def _frames_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f'No pude abrir {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    k = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        yield k / fps, img
        k += 1


def _frames_carpeta(path, fps):
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGENES)
    if not files:
        raise SystemExit(f'No hay imágenes en {path}')
    for k, p in enumerate(files):
        yield k / fps, cv2.imread(str(p))


def _frames_bag(path, topic):
    """rosbag2 sin cv_bridge (numpy 2 lo rompe en Humble): se decodifica a mano."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        raise SystemExit('Para leer un rosbag hace falta ROS: source /opt/ros/jazzy/setup.bash')
    meta = yaml.safe_load((path / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=meta['storage_identifier']),
                rosbag2_py.ConverterOptions('', ''))
    tipos = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in tipos:
        raise SystemExit(f'El bag no tiene {topic}. Tópicos: {sorted(tipos)}')
    msg_type = get_message(tipos[topic])
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    while reader.has_next():
        _, data, t_ns = reader.read_next()
        img = _decode(deserialize_message(data, msg_type))
        if img is not None:
            yield t_ns * 1e-9, img


def _decode(msg):
    if hasattr(msg, 'format'):                       # CompressedImage
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    enc = msg.encoding.lower()
    canales = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4, 'mono8': 1}.get(enc)
    if canales is None:
        print(f'Encoding {msg.encoding} no soportado, salto el frame')
        return None
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    img = img[:, :msg.width * canales].reshape(msg.height, msg.width, canales)
    conv = {'rgb8': cv2.COLOR_RGB2BGR, 'bgra8': cv2.COLOR_BGRA2BGR,
            'rgba8': cv2.COLOR_RGBA2BGR, 'mono8': cv2.COLOR_GRAY2BGR}.get(enc)
    return cv2.cvtColor(img, conv) if conv else img.copy()


# ---------------------------------------------------------------------- modelo
class Corredor:
    """Un modelo + su tracker, con los parámetros del YAML, como el nodo."""

    def __init__(self, nombre, pesos, cfg, a, PersonCounter):
        from ultralytics import YOLO
        self.nombre, self.pesos = nombre, pesos
        self.params, self.iou, names = cfg['params'], cfg['iou'], cfg['names']
        self.model = YOLO(resolver_pesos(pesos))
        self.classes = person_classes(self.model, names)
        self.device = a.device or None
        self.quant = fp16_kwargs(a.device)
        c = cfg['counting']
        max_missed = max(1, int(c.get('max_missed', 3)))
        if self.params.tiling and not a.tiles_siempre:
            max_missed *= cfg['every']
        self.counter = PersonCounter(min_hits=max(1, int(c.get('min_hits', 3))),
                                     max_age_sec=float(c.get('max_age_sec', 1.5)),
                                     max_missed=max_missed,
                                     iou_thr=float(c.get('iou_thr', 0.2)),
                                     dist_thr=float(c.get('dist_thr', 1.0)))
        self.por_frame, self.scores, self.ms, self.solo = [], [], [], 0
        self.n_cajas = 0

    def predict_fn(self, imgs, imgsz, conf):         # igual que PersonDetector._predict
        from person_yolo.inference import results_to_boxes
        res = self.model.predict(imgs, imgsz=imgsz, conf=conf, iou=self.iou,
                                 classes=self.classes, device=self.device, verbose=False,
                                 **self.quant)
        return [results_to_boxes(r) for r in res]


def leer_config(path, InferenceParams):
    params, iou, names = load_params(path, InferenceParams)
    p = yaml.safe_load(path.read_text())['person_detector']['ros__parameters']
    return {'params': params, 'iou': iou, 'names': names,
            'every': max(1, int(p.get('tiling', {}).get('every_n_frames', 1))),
            'counting': p.get('counting', {})}


# ---------------------------------------------------------------------- dibujo
def panel(frame, r, ids, c, ancho, draw_boxes):
    esc = ancho / frame.shape[1]
    img = cv2.resize(frame, (ancho, int(frame.shape[0] * esc)))
    draw_boxes(img, r.boxes, r.src, scale=esc, ids=ids)
    txt = (f'{c.nombre}: {len(r.boxes)} en cuadro | {c.counter.total} distintas | '
           f'{1000 * r.total_s:.0f} ms{" | tiles" if r.tiles else ""}')
    cv2.rectangle(img, (0, 0), (ancho, 30), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    return img


def mosaico(paneles):
    h, w = paneles[0].shape[:2]
    cols = 1 if len(paneles) == 1 else 2
    filas = -(-len(paneles) // cols)
    out = np.zeros((h * filas, w * cols, 3), np.uint8)
    for i, p in enumerate(paneles):
        f, c = divmod(i, cols)
        out[f * h:(f + 1) * h, c * w:(c + 1) * w] = p
    return out


# ------------------------------------------------------------------------ main
def main():
    a = parse_args()
    InferenceParams, detect_persons, _ = import_inference(REPO)
    from person_yolo.overlay import draw_boxes
    from person_yolo.tracker import PersonCounter

    cfg = leer_config(a.config.expanduser(), InferenceParams)
    print(f'Parámetros ({a.config.name}): {cfg["params"]}; tiles cada '
          f'{1 if a.tiles_siempre else cfg["every"]} frames')
    corredores = []
    for spec in a.model or MODELOS_POR_DEFECTO:
        nombre, _, pesos = spec.partition('=')
        if not pesos.endswith('.pt') or ('/' in pesos and not Path(pesos).exists()):
            raise SystemExit(f'No existe {pesos}')
        print(f'Cargando {nombre} ({pesos})')
        corredores.append(Corredor(nombre, pesos, cfg, a, PersonCounter))

    out = a.out / (a.src.expanduser().resolve().stem or 'fuente')
    (out / 'discrepancias').mkdir(parents=True, exist_ok=True)
    for viejo in (out / 'discrepancias').glob('*.jpg'):    # de una corrida anterior
        viejo.unlink()
    video = None
    candidatos = []                                  # (diferencia, k, mosaico)
    k = -1
    for k, (t, frame) in enumerate(frames(a)):
        use_tiles = a.tiles_siempre or k % cfg['every'] == 0
        resultados = [detect_persons(frame, c.predict_fn, c.params, use_tiles=use_tiles)
                      for c in corredores]
        if k == 0:                                   # el 1er frame calienta la GPU: no se mide
            resultados = [detect_persons(frame, c.predict_fn, c.params, use_tiles=use_tiles)
                          for c in corredores]

        paneles = []
        for i, (c, r) in enumerate(zip(corredores, resultados)):
            ids = c.counter.update(r.boxes, t)
            c.por_frame.append(len(r.boxes))
            c.scores.extend(r.boxes[:, 4].tolist())
            c.ms.append(1000 * r.total_s)
            c.n_cajas += len(r.boxes)
            # Cajas que ningún otro modelo ve (IoU < 0.3 con todas las de los demás).
            otras = [o.boxes[:, :4] for j, o in enumerate(resultados) if j != i]
            if len(r.boxes) and otras:
                otras = np.concatenate(otras)
                m = iou_matrix(r.boxes[:, :4], otras)
                c.solo += int((m.max(axis=1, initial=0) < 0.3).sum())
            paneles.append(panel(frame, r, ids, c, a.ancho_panel, draw_boxes))

        mos = mosaico(paneles)
        if video is None:
            tam = (mos.shape[1], mos.shape[0])
            video = cv2.VideoWriter(str(out / 'comparacion.mp4'),
                                    cv2.VideoWriter_fourcc(*'mp4v'), 10.0, tam)
        # VideoWriter descarta en silencio los frames de otro tamaño.
        video.write(mos if (mos.shape[1], mos.shape[0]) == tam else cv2.resize(mos, tam))

        conteos = [len(r.boxes) for r in resultados]
        dif = max(conteos) - min(conteos)
        if dif:
            candidatos.append((dif, k, mos))
            candidatos.sort(key=lambda x: -x[0])
            del candidatos[a.discrepancias:]
        if k % 50 == 0:
            print(f'  frame {k}: ' + ', '.join(f'{c.nombre}={n}'
                                               for c, n in zip(corredores, conteos)), flush=True)

    if video is None:
        raise SystemExit('La fuente no tiene frames.')
    video.release()
    for dif, kk, mos in candidatos:
        cv2.imwrite(str(out / 'discrepancias' / f'frame_{kk:05d}_dif{dif}.jpg'), mos,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    n = k + 1
    res = {c.nombre: {
        'pesos': c.pesos,
        'frames': n,
        'personas_por_frame': float(np.mean(c.por_frame)),
        'frames_con_personas': float(np.mean(np.array(c.por_frame) > 0)),
        'personas_distintas': c.counter.total,
        'cajas_solo_este_modelo': c.solo,
        'pct_solo_este_modelo': c.solo / max(c.n_cajas, 1),
        'score_medio': float(np.mean(c.scores)) if c.scores else 0.0,
        'ms_por_frame': float(np.mean(c.ms)),
        'ms_p95': float(np.percentile(c.ms, 95)),
    } for c in corredores}

    l = [f'## Comparación en {a.src} ({n} frames, {a.config.name}, tiles cada '
         f'{1 if a.tiles_siempre else cfg["every"]})', '',
         '| Modelo | Personas/frame | Frames con personas | Personas distintas '
         '| Cajas que solo ve él | Score medio | ms/frame | ms p95 |',
         '|---|---|---|---|---|---|---|---|']
    for nombre, r in res.items():
        l.append(f'| {nombre} | {r["personas_por_frame"]:.2f} | {r["frames_con_personas"]:.0%} '
                 f'| {r["personas_distintas"]} | {r["cajas_solo_este_modelo"]} '
                 f'({r["pct_solo_este_modelo"]:.0%}) | {r["score_medio"]:.2f} '
                 f'| {r["ms_por_frame"]:.0f} | {r["ms_p95"]:.0f} |')
    l += ['', 'Sin etiquetas: más cajas o más personas distintas NO es mejor si son falsas '
          'alarmas. "Cajas que solo ve él" = sin ninguna caja de otro modelo con IoU >= 0.3: '
          'son hallazgos propios o falsas alarmas; mirar discrepancias/ y comparacion.mp4. '
          'ms medidos en esta GPU (con Unreal corriendo serían más).']
    md = '\n'.join(l) + '\n'
    (out / 'resumen.md').write_text(md)
    (out / 'resumen.json').write_text(json.dumps(res, indent=2))
    print('\n' + md + f'\nSalida en {out}')


if __name__ == '__main__':
    main()
