"""Inferencia en mosaico (tiling) para detectar personas lejanas.

Una persona a 80 m ocupa pocos píxeles. Si el frame de 1920x1080 se achica a
640 para el modelo, esa persona queda en 3-4 px y desaparece. Cortando el frame
en tiles de 640 a resolución nativa, cada tile entra al modelo SIN achicarse y
la persona conserva sus píxeles. Las cercanas las sigue agarrando la pasada de
frame completo, que ve el contexto entero (una persona grande partida entre
tiles queda truncada en cada uno).

Python puro (solo numpy): sin rclpy ni ultralytics, testeable con pytest.

Formato de cajas en todo el módulo: ndarray (N, 6) con columnas
    x1, y1, x2, y2, score, class_id
en píxeles del frame completo (o del tile, antes de remapear).
"""
import numpy as np

BOX_COLS = 6


def _axis_starts(length: int, tile: int, step: int):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)              # último tile pegado al borde
    return starts


def make_tiles(width: int, height: int, tile: int, overlap: float):
    """Lista de tiles (x0, y0, x1, y1) que cubren la imagen con solape.

    Todos los tiles miden `tile` x `tile` (salvo que la imagen sea más chica
    en ese eje); el último de cada fila/columna se corre para quedar pegado al
    borde en vez de salirse o quedar parcial. Si la imagen entera ya cabe en un
    tile devuelve un solo tile = la imagen: el tiling no aportaría nada.
    """
    if tile <= 0:
        raise ValueError('tile debe ser > 0')
    if not 0.0 <= overlap < 1.0:
        raise ValueError('overlap debe estar en [0, 1)')
    step = max(1, int(round(tile * (1.0 - overlap))))
    xs = _axis_starts(width, tile, step)
    ys = _axis_starts(height, tile, step)
    return [(x, y, min(x + tile, width), min(y + tile, height)) for y in ys for x in xs]


def remap_tile_boxes(boxes, tile_rect, width: int, height: int, edge_px: float = 3.0):
    """Pasa cajas de coordenadas del tile a coordenadas del frame.

    Descarta las cajas que tocan un borde INTERNO del tile (uno que no es
    borde de la imagen): ahí el objeto está cortado, la caja sale truncada y
    con score bajo. Ese objeto lo cubre otro tile (por el solape) o la pasada
    de frame completo. Las que tocan un borde real de la imagen se conservan.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, BOX_COLS)
    if not len(boxes):
        return boxes
    x0, y0, x1, y1 = tile_rect
    keep = np.ones(len(boxes), dtype=bool)
    if x0 > 0:
        keep &= boxes[:, 0] > edge_px
    if y0 > 0:
        keep &= boxes[:, 1] > edge_px
    if x1 < width:
        keep &= boxes[:, 2] < (x1 - x0) - edge_px
    if y1 < height:
        keep &= boxes[:, 3] < (y1 - y0) - edge_px
    out = boxes[keep].copy()
    out[:, [0, 2]] += x0
    out[:, [1, 3]] += y0
    return out


def _overlap(box, others, metric: str):
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    if metric == 'ios':
        denom = np.minimum(area, areas)               # intersección / caja menor
    else:
        denom = area + areas - inter                  # IoU clásico
    return inter / np.maximum(denom, 1e-9)


def nms(boxes, thr: float = 0.5, metric: str = 'iou'):
    """NMS por clase. Devuelve los índices conservados, mayor score primero.

    `metric='ios'` (intersección sobre la caja menor) además elimina una caja
    chica contenida en otra grande — útil cuando la misma persona aparece
    entera en el frame completo y recortada en un tile.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, BOX_COLS)
    keep = []
    for cls in np.unique(boxes[:, 5]):
        idx = np.where(boxes[:, 5] == cls)[0]
        idx = idx[np.argsort(-boxes[idx, 4], kind='stable')]
        while len(idx):
            best = idx[0]
            keep.append(best)
            if len(idx) == 1:
                break
            ov = _overlap(boxes[best], boxes[idx[1:]], metric)
            idx = idx[1:][ov < thr]
    keep.sort(key=lambda i: -boxes[i, 4])
    return keep


def merge(full_boxes, tile_results, width: int, height: int,
          thr: float = 0.5, metric: str = 'iou', edge_px: float = 3.0):
    """Une detecciones de frame completo y de tiles en una sola lista.

    `tile_results` es una lista de (tile_rect, boxes_en_coords_del_tile).
    Devuelve (boxes, source) donde source[i] es 0 si la caja salió del frame
    completo y 1 si salió de un tile (para dibujarlas en otro color).
    """
    parts = [np.asarray(full_boxes, dtype=np.float64).reshape(-1, BOX_COLS)]
    sources = [np.zeros(len(parts[0]), dtype=np.int8)]
    for rect, boxes in tile_results:
        remapped = remap_tile_boxes(boxes, rect, width, height, edge_px)
        parts.append(remapped)
        sources.append(np.ones(len(remapped), dtype=np.int8))
    allb = np.concatenate(parts, axis=0)
    src = np.concatenate(sources)
    if not len(allb):
        return allb, src
    keep = nms(allb, thr, metric)
    return allb[keep], src[keep]
