#!/usr/bin/env python3
"""Etapa 4: comparar modelos en VisDrone test-dev (test nunca se usó para decidir nada).

Dos niveles, para cada modelo:

1. MODELO, sobre los tiles de test (Ultralytics val): precisión, recall, mAP50,
   mAP50-95, curvas PR y matriz de confusión.

2. SISTEMA, sobre las imágenes COMPLETAS de test-dev, pasando cada una por
   detect_persons() de inference.py (la misma función que usa el nodo en vuelo)
   con los parámetros de jetson.yaml: frame completo + tiles + merge, a los
   umbrales reales (conf y tiling.conf). Mide:
     - precisión, recall y F1 a esos umbrales
     - recall por tamaño de persona (lado menor < 16 px, 16-32 px, >= 32 px)
     - falsas alarmas por imagen y ms por imagen (en ESTA GPU, no en la Jetson)
   y guarda imágenes de ejemplo: verde = encontrada, rojo = perdida,
   amarillo = falsa alarma.

Una detección es correcta si su IoU con una persona etiquetada es >= --iou
(0.5, el estándar de VisDrone). Las detecciones dentro de regiones ignoradas
de VisDrone no cuentan como falsas alarmas: ahí hay personas sin etiquetar.

Limitación: en vuelo los tiles corren cada `every_n_frames` frames; acá corren
en todas las imágenes. Esto mide el mejor caso por imagen.

Uso:
  python evaluate.py --repo ~/UAV_lab/YOLO \\
      --data ~/datasets/visdrone_tiles/data.yaml --src ~/datasets/VisDrone \\
      --model coco=yolo11n.pt \\
      --model n_sin_tuning=runs/train/n_sin_tuning/weights/best.pt \\
      --model n_con_tuning=runs/train/n_con_tuning/weights/best.pt \\
      --model s_con_tuning=runs/train/s_con_tuning/weights/best.pt
Por defecto el nivel 2 se mide con sitl.yaml (lo que corre hoy) y con jetson.yaml
(el dron real); --config permite elegir otros.
Salida (en runs/eval/): resumen.md, resumen.json, val_<modelo>/ y
ejemplos_<config>_<modelo>/.
Si junto a los pesos está la ficha_modelo.json de train.py, completa `metricas_test`.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Reutiliza la lectura de anotaciones de la etapa 1: mismas clases, mismas regiones ignoradas.
from prepare_visdrone import read_annotations

TAMANOS = ('<16px', '16-32px', '>=32px')
VERDE, ROJO, AMARILLO = (0, 200, 0), (0, 0, 255), (0, 220, 255)


def parse_args():
    ap = argparse.ArgumentParser(description='Evaluación y comparación de modelos (etapa 4).')
    ap.add_argument('--repo', required=True, type=Path, help='raíz del repo yolo')
    ap.add_argument('--data', required=True, type=Path, help='data.yaml de los tiles')
    ap.add_argument('--src', required=True, type=Path, help='carpeta con VisDrone2019-DET-test-dev')
    ap.add_argument('--model', action='append', required=True, metavar='NOMBRE=PESOS',
                    help='repetir por modelo, p. ej. coco=yolo11n.pt')
    ap.add_argument('--config', type=Path, action='append',
                    help='YAML del detector; repetir para varios (por defecto sitl.yaml '
                         'y jetson.yaml del repo)')
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--device', default='', help="'' = automático, 'cpu', '0'...")
    ap.add_argument('--batch', type=int, default=8, help='batch de la validación por tiles')
    ap.add_argument('--ejemplos', type=int, default=6, help='imágenes de ejemplo por modelo')
    ap.add_argument('--limit', type=int, default=0, help='solo N imágenes completas (pruebas)')
    ap.add_argument('--out', type=Path, default=Path('runs/eval'))
    return ap.parse_args()


# ----------------------------------------------------------------- preparación
def import_inference(repo: Path):
    pkg = repo / 'src' / 'person_yolo'
    if not (pkg / 'person_yolo' / 'inference.py').exists():
        sys.exit(f'No encuentro {pkg}/person_yolo/inference.py. ¿Ya hiciste merge del PR #1 '
                 '(o estás en la rama refactor/inferencia-pura)?')
    sys.path.insert(0, str(pkg))
    from person_yolo.inference import InferenceParams, detect_persons, results_to_boxes
    return InferenceParams, detect_persons, results_to_boxes


def load_params(config: Path, InferenceParams):
    """Lee los parámetros del detector desde el MISMO YAML que usa el nodo."""
    p = yaml.safe_load(config.read_text())['person_detector']['ros__parameters']
    t = p.get('tiling', {})
    params = InferenceParams(
        imgsz=p['imgsz'], conf=p['conf'], tiling=t.get('enabled', True),
        tile_size=t['tile_size'], tile_overlap=t['overlap'], tile_conf=t['conf'],
        tile_batch=t.get('batch_size', 0), merge_thr=t.get('merge_thr', 0.5),
        merge_metric=t.get('merge_metric', 'ios'))
    return params, p.get('iou', 0.45), p.get('person_class_names', ['person'])


def person_classes(model, names_wanted):
    """Ids de las clases de persona, buscadas por nombre como hace el nodo."""
    ids = [i for i, n in model.names.items() if n in names_wanted]
    if not ids:
        sys.exit(f'El modelo no tiene clases {names_wanted}: {model.names}')
    return ids


# --------------------------------------------------------- nivel 1: modelo
def eval_tiles(model, nombre, a, classes):
    m = model.val(data=str(a.data), split='test', imgsz=640, batch=a.batch, iou=0.7,
                  classes=classes, device=a.device or None, plots=True, verbose=False,
                  project=str(a.out.resolve()), name=f'val_{nombre}', exist_ok=True)
    r = m.results_dict
    return {'precision': r['metrics/precision(B)'], 'recall': r['metrics/recall(B)'],
            'mAP50': r['metrics/mAP50(B)'], 'mAP50-95': r['metrics/mAP50-95(B)']}


# --------------------------------------------------------- nivel 2: sistema
def iou_matrix(a, b):
    """IoU entre cada caja de `a` (N, 4) y cada una de `b` (M, 4)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match(dets, gts, ignored, iou_thr):
    """Empareja detecciones con personas etiquetadas, de mayor a menor score.

    Devuelve (gt_encontrada: bool por persona, tipo por detección) con tipo en
    {'tp', 'fp', 'ign'}. 'ign' = falsa alarma dentro de una región ignorada.
    """
    order = np.argsort(-dets[:, 4]) if len(dets) else []
    ious = iou_matrix(dets[:, :4], gts)
    found = np.zeros(len(gts), dtype=bool)
    tipo = ['fp'] * len(dets)
    for i in order:
        if len(gts):
            cand = np.where(~found & (ious[i] >= iou_thr))[0]
            if len(cand):
                found[cand[np.argmax(ious[i, cand])]] = True
                tipo[i] = 'tp'
                continue
        cx, cy = (dets[i, 0] + dets[i, 2]) / 2, (dets[i, 1] + dets[i, 3]) / 2
        if any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in ignored):
            tipo[i] = 'ign'
    return found, tipo


def size_of(box):
    side = min(box[2] - box[0], box[3] - box[1])
    return TAMANOS[0] if side < 16 else TAMANOS[1] if side < 32 else TAMANOS[2]


def draw(frame, gts, found, dets, tipo):
    img = frame.copy()
    for g, f in zip(gts, found):
        cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])),
                      VERDE if f else ROJO, 2)
    for d, t in zip(dets, tipo):
        if t == 'fp':
            cv2.rectangle(img, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), AMARILLO, 2)
    return img


def eval_system(model, nombre, a, images, ejemplos, params, iou_nms, classes, fns, cfg):
    _, detect_persons, results_to_boxes = fns

    def predict_fn(imgs, imgsz, conf):        # igual que PersonDetector._predict
        res = model.predict(imgs, imgsz=imgsz, conf=conf, iou=iou_nms, classes=classes,
                            device=a.device or None, half=True, verbose=False)
        return [results_to_boxes(r) for r in res]

    detect_persons(cv2.imread(str(images[0])), predict_fn, params)    # calentar la GPU
    tp = fp = n_gt = 0
    gt_size, found_size = dict.fromkeys(TAMANOS, 0), dict.fromkeys(TAMANOS, 0)
    tiempos = []
    out_ej = a.out / f'ejemplos_{cfg}_{nombre}'
    out_ej.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        frame = cv2.imread(str(img_path))
        persons, ignored = read_annotations(img_path.parent.parent / 'annotations'
                                            / f'{img_path.stem}.txt')
        gts = np.array(persons, dtype=float).reshape(-1, 4)
        r = detect_persons(frame, predict_fn, params, use_tiles=True)
        tiempos.append(r.total_s)

        found, tipo = match(r.boxes, gts, ignored, a.iou)
        tp += tipo.count('tp')
        fp += tipo.count('fp')
        n_gt += len(gts)
        for g, f in zip(gts, found):
            gt_size[size_of(g)] += 1
            found_size[size_of(g)] += int(f)
        if img_path in ejemplos:
            cv2.imwrite(str(out_ej / f'{img_path.stem}.jpg'), draw(frame, gts, found, r.boxes, tipo),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

    p = tp / max(tp + fp, 1)
    rc = tp / max(n_gt, 1)
    return {
        'precision': p, 'recall': rc, 'F1': 2 * p * rc / max(p + rc, 1e-9),
        'recall_por_tamano': {k: found_size[k] / max(gt_size[k], 1) for k in TAMANOS},
        'personas_por_tamano': gt_size,
        'falsas_alarmas_por_imagen': fp / len(images),
        'ms_por_imagen': 1000 * float(np.mean(tiempos)),
        'imagenes': len(images),
    }


# ------------------------------------------------------------------- resumen
def resumen_md(res, configs, a):
    f = '{:.3f}'.format
    l = ['## Nivel 1 — modelo (tiles de test)', '',
         '| Modelo | Precisión | Recall | mAP50 | mAP50-95 |', '|---|---|---|---|---|']
    for n, r in res.items():
        t = r['tiles']
        l.append(f'| {n} | {f(t["precision"])} | {f(t["recall"])} | {f(t["mAP50"])} '
                 f'| {f(t["mAP50-95"])} |')
    # Una tabla del sistema por cada configuración del detector (sitl, jetson...).
    for cfg, (params, _, _) in configs.items():
        l += ['', f'## Nivel 2 — sistema con {cfg}.yaml (imágenes completas, conf '
              f'{params.conf} / tiles {params.tile_conf}, overlap {params.tile_overlap}, '
              f'IoU {a.iou})', '',
              '| Modelo | Precisión | Recall | F1 | Recall <16px | Recall 16-32px '
              '| Recall >=32px | Falsas alarmas/img | ms/img |',
              '|---|---|---|---|---|---|---|---|---|']
        for n, r in res.items():
            s = r['sistema'][cfg]
            rt = s['recall_por_tamano']
            l.append(f'| {n} | {f(s["precision"])} | {f(s["recall"])} | {f(s["F1"])} '
                     f'| {f(rt["<16px"])} | {f(rt["16-32px"])} | {f(rt[">=32px"])} '
                     f'| {s["falsas_alarmas_por_imagen"]:.2f} | {s["ms_por_imagen"]:.0f} |')
    primero = next(iter(next(iter(res.values()))['sistema'].values()))
    l += ['', f'Personas en test por tamaño: {primero["personas_por_tamano"]} '
          f'({primero["imagenes"]} imágenes).',
          'Nivel 1: precisión y recall al umbral que maximiza F1 (Ultralytics). '
          'Nivel 2: a los umbrales reales del detector. ms medidos en esta GPU, no en la Jetson. '
          'Los tiles corren en todas las imágenes (en vuelo, cada every_n_frames).']
    return '\n'.join(l) + '\n'


def update_ficha(weights: Path, r):
    ficha = weights.parent.parent / 'ficha_modelo.json'
    if ficha.exists():
        d = json.loads(ficha.read_text())
        d['metricas_test'] = r
        ficha.write_text(json.dumps(d, indent=2, ensure_ascii=False))


def main():
    a = parse_args()
    a.repo, a.data, a.src = a.repo.expanduser(), a.data.expanduser(), a.src.expanduser()
    a.out.mkdir(parents=True, exist_ok=True)
    fns = import_inference(a.repo)
    cfg_dir = a.repo / 'src' / 'person_yolo' / 'config'
    rutas = [p.expanduser() for p in a.config] if a.config else \
        [cfg_dir / 'sitl.yaml', cfg_dir / 'jetson.yaml']
    configs = {}                                   # 'sitl' -> (params, iou, nombres de clase)
    for ruta in rutas:
        configs[ruta.stem] = load_params(ruta, fns[0])
        print(f'Parámetros del detector ({ruta.name}): {configs[ruta.stem][0]}')
    names_wanted = sorted({n for _, _, ns in configs.values() for n in ns})

    images = sorted((a.src / 'VisDrone2019-DET-test-dev' / 'images').glob('*.jpg'))
    if a.limit:
        images = images[:a.limit]
    if not images:
        sys.exit(f'No hay imágenes en {a.src}/VisDrone2019-DET-test-dev/images')
    # Las MISMAS imágenes de ejemplo para todos los modelos, para compararlas lado a lado.
    ejemplos = set(random.Random(0).sample(images, min(a.ejemplos, len(images))))

    from ultralytics import YOLO
    res = {}
    for spec in a.model:
        nombre, _, pesos = spec.partition('=')
        print(f'\n=== {nombre} ({pesos}) ===')
        model = YOLO(pesos)
        classes = person_classes(model, names_wanted)
        res[nombre] = {'pesos': pesos,
                       # Nivel 1 no depende del YAML del detector: se calcula una vez.
                       'tiles': eval_tiles(model, nombre, a, classes),
                       'sistema': {cfg: eval_system(model, nombre, a, images, ejemplos,
                                                    params, iou_nms, classes, fns, cfg)
                                   for cfg, (params, iou_nms, _) in configs.items()}}
        print(json.dumps(res[nombre], indent=2))
        update_ficha(Path(pesos), {k: res[nombre][k] for k in ('tiles', 'sistema')})

    md = resumen_md(res, configs, a)
    (a.out / 'resumen.md').write_text(md)
    (a.out / 'resumen.json').write_text(json.dumps(res, indent=2))
    print('\n' + md)


if __name__ == '__main__':
    main()
