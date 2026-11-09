#!/usr/bin/env python3
"""Etapa 1: VisDrone2019-DET -> dataset YOLO de tiles de 640 con una sola clase, `person`.

Por qué tiles: el detector en vuelo (person_yolo/inference.py) busca personas
lejanas en recortes de 640 a RESOLUCIÓN NATIVA. Si entrenáramos con la imagen
completa achicada a 640, el modelo vería a las personas ~2-3 veces más chicas
que en vuelo. Cortando con el mismo make_tiles() del repo, entrenamiento e
inferencia ven a las personas del mismo tamaño.

Decisiones (ver plan):
  - Clases: pedestrian (1) y people (2) de VisDrone -> `person` (0). Lo demás se descarta.
  - Personas cortadas por el borde del tile: se etiquetan (caja recortada al tile).
    Si el corte deja una astilla de menos de --min-px de ancho o alto, se descarta.
  - Regiones ignoradas de VisDrone (categoría 0: multitudes sin etiquetar persona
    por persona): se tapan con gris. Si no, el modelo vería personas sin etiqueta
    y aprendería que son fondo.
  - Tiles sin personas: en train se guarda una muestra para que sean --bg-ratio
    del total. En val y test se guardan TODOS: ahí se mide, y la proporción de
    fondo tiene que ser la real o la precisión saldría inflada.
  - Las divisiones oficiales no se mezclan: los tiles de una imagen quedan siempre
    en la división de esa imagen.

Uso:
  python prepare_visdrone.py --repo ~/UAV_lab/YOLO --src ~/datasets/VisDrone --out ~/datasets/visdrone_tiles --download
"""
import argparse
import json
import random
import sys
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.request import urlretrieve

import cv2
from PIL import Image

# División oficial de VisDrone -> nombre de la división en el dataset de salida.
SPLITS = {
    'VisDrone2019-DET-train': 'train',
    'VisDrone2019-DET-val': 'val',
    'VisDrone2019-DET-test-dev': 'test',
}
URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/{}.zip'
PERSON_CATS = {1, 2}        # 1 = pedestrian (de pie o caminando), 2 = people (otra postura)
IGNORED_CAT = 0             # regiones ignoradas
GRAY = (114, 114, 114)      # el mismo gris que usa Ultralytics para rellenar


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--repo', required=True, type=Path,
                    help='raíz del repo yolo (de ahí se importa tiling.py)')
    ap.add_argument('--src', required=True, type=Path,
                    help='carpeta con las divisiones VisDrone2019-DET-* originales')
    ap.add_argument('--out', required=True, type=Path, help='carpeta del dataset de salida')
    ap.add_argument('--download', action='store_true', help='bajar las divisiones que falten')
    ap.add_argument('--tile', type=int, default=640, help='= tiling.tile_size del YAML')
    ap.add_argument('--overlap', type=float, default=0.15, help='= tiling.overlap de jetson.yaml')
    ap.add_argument('--min-px', type=int, default=4, help='astilla mínima de una persona cortada')
    ap.add_argument('--bg-ratio', type=float, default=0.10, help='fracción de tiles vacíos en train')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='procesar solo N imágenes por división (pruebas)')
    return ap.parse_args()


def import_make_tiles(repo: Path):
    """Importa make_tiles() del repo: el MISMO criterio de corte que usa el detector."""
    pkg = repo / 'src' / 'person_yolo'
    if not (pkg / 'person_yolo' / 'tiling.py').exists():
        sys.exit(f'No encuentro {pkg}/person_yolo/tiling.py. ¿--repo apunta a la raíz del repo?')
    sys.path.insert(0, str(pkg))
    from person_yolo.tiling import make_tiles
    return make_tiles


def ensure_split(src: Path, name: str, download: bool):
    d = src / name
    if (d / 'images').is_dir() and (d / 'annotations').is_dir():
        return d
    if not download:
        sys.exit(f'Falta {d}. Bajalo o corré con --download.')
    src.mkdir(parents=True, exist_ok=True)
    zpath = src / f'{name}.zip'
    print(f'Bajando {name}...')
    urlretrieve(URL.format(name), zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(src)
    zpath.unlink()
    return d


def read_annotations(path: Path):
    """Formato VisDrone: x, y, w, h, score, categoría, truncamiento, oclusión (píxeles)."""
    persons, ignored = [], []
    for line in path.read_text().splitlines():
        p = line.strip().rstrip(',').split(',')
        if len(p) < 6:
            continue
        x, y, w, h, _score, cat = map(int, p[:6])
        if w <= 0 or h <= 0:
            continue
        box = (x, y, x + w, y + h)                  # a x1, y1, x2, y2
        if cat in PERSON_CATS:
            persons.append(box)
        elif cat == IGNORED_CAT:
            ignored.append(box)
    return persons, ignored


def clip(box, rect):
    """Intersección de una caja con un tile, en coordenadas del tile (o None)."""
    x0, y0, x1, y1 = rect
    a, b = max(box[0], x0), max(box[1], y0)
    c, d = min(box[2], x1), min(box[3], y1)
    if c <= a or d <= b:
        return None
    return (a - x0, b - y0, c - x0, d - y0)


def tile_labels(persons, rect, min_px):
    """Cajas de las personas dentro del tile. Devuelve (cajas, astillas descartadas)."""
    out, slivers = [], 0
    for box in persons:
        c = clip(box, rect)
        if c is None:
            continue
        cut = (c[2] - c[0], c[3] - c[1]) != (box[2] - box[0], box[3] - box[1])
        # La regla de la astilla es solo para personas CORTADAS: una persona
        # entera de 3 px es real y se queda.
        if cut and (c[2] - c[0] < min_px or c[3] - c[1] < min_px):
            slivers += 1
            continue
        out.append(c)
    return out, slivers


def plan_split(split_dir, make_tiles, args):
    """Primera pasada (solo anotaciones): qué tiles hay y cuáles tienen personas."""
    images = sorted((split_dir / 'images').glob('*.jpg'))
    if args.limit:
        images = images[:args.limit]
    plan, slivers = [], 0
    for img in images:
        persons, ignored = read_annotations(split_dir / 'annotations' / f'{img.stem}.txt')
        w, h = Image.open(img).size                 # lee solo el encabezado
        for rect in make_tiles(w, h, args.tile, args.overlap):
            boxes, s = tile_labels(persons, rect, args.min_px)
            slivers += s
            plan.append({'img': str(img), 'rect': rect, 'boxes': boxes,
                         'ignored': [c for c in (clip(b, rect) for b in ignored) if c]})
    return plan, len(images), slivers


def choose_backgrounds(plan, ratio, rng):
    """Tiles vacíos a conservar para que sean `ratio` del total: B = P * r / (1 - r)."""
    pos = [t for t in plan if t['boxes']]
    neg = [t for t in plan if not t['boxes']]
    n_bg = min(len(neg), round(len(pos) * ratio / (1.0 - ratio)))
    return pos + rng.sample(neg, n_bg)


def write_image_tiles(job):
    """Segunda pasada, por imagen: recorta, tapa lo ignorado y escribe imagen + etiqueta."""
    img_path, tiles, img_dir, lbl_dir = job
    frame = cv2.imread(img_path)
    stem = Path(img_path).stem
    for t in tiles:
        x0, y0, x1, y1 = t['rect']
        crop = frame[y0:y1, x0:x1].copy()
        for a, b, c, d in t['ignored']:
            crop[b:d, a:c] = GRAY
        name = f'{stem}_x{x0}_y{y0}'
        cv2.imwrite(str(Path(img_dir) / f'{name}.jpg'), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        tw, th = x1 - x0, y1 - y0
        # Formato YOLO: clase cx cy w h, normalizado por el tamaño del tile.
        lines = [f'0 {(a + c) / 2 / tw:.6f} {(b + d) / 2 / th:.6f} '
                 f'{(c - a) / tw:.6f} {(d - b) / th:.6f}' for a, b, c, d in t['boxes']]
        (Path(lbl_dir) / f'{name}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
    return len(tiles)


def size_bucket(box):
    side = min(box[2] - box[0], box[3] - box[1])
    return '<16px' if side < 16 else '16-32px' if side < 32 else '>=32px'


def main():
    args = parse_args()
    make_tiles = import_make_tiles(args.repo.expanduser())
    src, out = args.src.expanduser(), args.out.expanduser()
    rng = random.Random(args.seed)
    stats = {'params': {k: str(v) for k, v in vars(args).items()}, 'splits': {}}

    for vis_name, split in SPLITS.items():
        split_dir = ensure_split(src, vis_name, args.download)
        plan, n_images, slivers = plan_split(split_dir, make_tiles, args)
        kept = choose_backgrounds(plan, args.bg_ratio, rng) if split == 'train' else plan

        img_dir, lbl_dir = out / 'images' / split, out / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        by_image = {}
        for t in kept:
            by_image.setdefault(t['img'], []).append(t)
        jobs = [(img, tiles, str(img_dir), str(lbl_dir)) for img, tiles in by_image.items()]
        with ProcessPoolExecutor(args.workers) as ex:
            n_written = sum(ex.map(write_image_tiles, jobs, chunksize=8))

        sizes = Counter(size_bucket(b) for t in kept for b in t['boxes'])
        s = stats['splits'][split] = {
            'imagenes': n_images,
            'tiles_totales': len(plan),
            'tiles_guardados': n_written,
            'tiles_con_personas': sum(1 for t in kept if t['boxes']),
            'tiles_vacios': sum(1 for t in kept if not t['boxes']),
            'personas': sum(len(t['boxes']) for t in kept),
            'astillas_descartadas': slivers,
            'personas_por_tamano': dict(sorted(sizes.items())),
        }
        print(f'{split}: {s}')

    # data.yaml para Ultralytics. `test` solo se usa en la evaluación final.
    (out / 'data.yaml').write_text(
        f'path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n'
        'names:\n  0: person\n')
    (out / 'stats.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f'Listo: {out / "data.yaml"}')


if __name__ == '__main__':
    main()
