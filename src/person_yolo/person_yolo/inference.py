"""Inferencia de personas: frame completo + tiles, fusionado con NMS.

Es la MISMA función que usa el nodo (person_detector.py) y cualquier script
que quiera medir el detector (evaluación, benchmarks). Si la evaluación
reescribiera esta lógica por su cuenta, mediría una copia y no lo que vuela.

Python puro: sin rclpy ni ultralytics, testeable con pytest. El modelo entra
como `predict_fn`, una función que recibe una lista de imágenes y devuelve una
lista de cajas; el nodo le pasa una que llama a YOLO y los tests una falsa.

Formato de cajas: ndarray (N, 6) x1, y1, x2, y2, score, class_id (ver tiling.py).
"""
import time
from dataclasses import dataclass, field

import numpy as np

from person_yolo.tiling import BOX_COLS, make_tiles, merge


@dataclass(frozen=True)
class InferenceParams:
    """Parámetros de la inferencia (sección `person_detector` del YAML)."""
    imgsz: int = 640              # pasada de frame completo
    conf: float = 0.35            # umbral de la pasada de frame completo
    tiling: bool = True
    tile_size: int = 640
    tile_overlap: float = 0.2
    tile_conf: float = 0.25       # más bajo: una persona lejana da menos score
    tile_batch: int = 0           # 0 = todos los tiles en un solo batch
    merge_thr: float = 0.5
    merge_metric: str = 'ios'     # iou | ios


@dataclass
class InferenceResult:
    boxes: np.ndarray             # (N, 6) en píxeles del frame completo
    src: np.ndarray               # (N,) 0 = frame completo, 1 = tile
    tiles: list = field(default_factory=list)   # tiles usados (vacío si no hubo)
    too_small: bool = False       # se pidieron tiles pero la imagen cabe en uno
    full_s: float = 0.0           # segundos de la pasada de frame completo
    tiles_s: float = 0.0          # segundos de los tiles (0 si no hubo)
    total_s: float = 0.0          # todo, incluido el merge


def results_to_boxes(results):
    """Results de ultralytics -> ndarray (N, 6) x1,y1,x2,y2,score,cls."""
    b = results.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, BOX_COLS))
    return np.concatenate([
        b.xyxy.cpu().numpy(),
        b.conf.cpu().numpy().reshape(-1, 1),
        b.cls.cpu().numpy().reshape(-1, 1),
    ], axis=1).astype(np.float64)


def detect_persons(frame, predict_fn, params: InferenceParams, use_tiles: bool = True):
    """Detecta personas en un frame (ndarray HxWxC).

    `predict_fn(images, imgsz, conf)` recibe una LISTA de imágenes y devuelve una
    lista del mismo largo con las cajas (N, 6) de cada una.

    `use_tiles` lo decide quien llama: el nodo solo cada `every_n_frames`
    frames (para ahorrar GPU), una evaluación en todas las imágenes. Igual no
    hay tiles si `params.tiling` está apagado o si la imagen cabe en un tile
    (en ese caso `too_small` queda en True para que quien llama avise).
    """
    h, w = frame.shape[:2]
    t0 = time.monotonic()
    full = predict_fn([frame], params.imgsz, params.conf)[0]
    t1 = time.monotonic()

    tiles = []
    tile_results = []
    too_small = False
    tiles_s = 0.0
    if use_tiles and params.tiling:
        tiles = make_tiles(w, h, params.tile_size, params.tile_overlap)
        if len(tiles) <= 1:
            too_small = True          # el tiling no aportaría nada
            tiles = []
        else:
            crops = [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in tiles]
            bs = params.tile_batch if params.tile_batch > 0 else len(crops)
            for i in range(0, len(crops), bs):
                res = predict_fn(crops[i:i + bs], params.tile_size, params.tile_conf)
                tile_results.extend(zip(tiles[i:i + bs], res))
            tiles_s = time.monotonic() - t1

    # Las de tile pasaron con tile_conf (más bajo); las del frame completo con
    # conf. Se conserva todo lo que sobrevive al NMS.
    boxes, src = merge(full, tile_results, w, h, params.merge_thr, params.merge_metric)
    return InferenceResult(boxes=boxes, src=src, tiles=tiles, too_small=too_small,
                           full_s=t1 - t0, tiles_s=tiles_s,
                           total_s=time.monotonic() - t0)
