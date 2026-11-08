import numpy as np
import pytest

from person_yolo.inference import InferenceParams, detect_persons, results_to_boxes


class FakeModel:
    """predict_fn falso: devuelve cajas fijas según el tamaño de la imagen.

    - Imagen completa (más grande que un tile): una persona cercana en (100, 100).
    - Tile en la esquina (0, 0): una persona lejana chica en (300, 300) del tile.
    Registra cada llamada para revisar imgsz, conf y tamaño de los batches.
    """

    def __init__(self, tile_size=640):
        self.tile_size = tile_size
        self.calls = []

    def __call__(self, images, imgsz, conf):
        self.calls.append({'n': len(images), 'imgsz': imgsz, 'conf': conf})
        out = []
        for img in images:
            h, w = img.shape[:2]
            if w > self.tile_size or h > self.tile_size:
                out.append(np.array([[100, 100, 180, 300, 0.9, 0]], dtype=float))
            elif img[0, 0, 0] == 7:       # marca puesta en la esquina del frame
                out.append(np.array([[300, 300, 312, 324, 0.4, 0]], dtype=float))
            else:
                out.append(np.zeros((0, 6)))
        return out


def _frame(w=1920, h=1080):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[0, 0, 0] = 7                    # solo el tile (0, 0) la ve
    return frame


def test_full_frame_and_tiles_are_merged():
    model = FakeModel()
    r = detect_persons(_frame(), model, InferenceParams(tile_overlap=0.2))
    assert len(r.tiles) == 8              # 1080p con tiles de 640 y 20 % de solape
    assert len(r.boxes) == 2
    # La lejana viene del tile (0, 0): sus coordenadas no se mueven al remapear.
    far = r.boxes[r.src == 1]
    assert far.shape == (1, 6)
    np.testing.assert_allclose(far[0, :4], [300, 300, 312, 324])
    assert (r.src == 0).sum() == 1


def test_thresholds_and_sizes_reach_the_model():
    model = FakeModel()
    params = InferenceParams(imgsz=640, conf=0.35, tile_size=640, tile_conf=0.25,
                             tile_overlap=0.2)
    detect_persons(_frame(), model, params)
    full, tiles = model.calls[0], model.calls[1]
    assert (full['n'], full['imgsz'], full['conf']) == (1, 640, 0.35)
    assert (tiles['n'], tiles['imgsz'], tiles['conf']) == (8, 640, 0.25)


def test_tile_batch_splits_calls():
    model = FakeModel()
    detect_persons(_frame(), model, InferenceParams(tile_overlap=0.2, tile_batch=3))
    assert [c['n'] for c in model.calls] == [1, 3, 3, 2]


@pytest.mark.parametrize('use_tiles,tiling', [(False, True), (True, False)])
def test_no_tiles_when_not_requested_or_disabled(use_tiles, tiling):
    model = FakeModel()
    r = detect_persons(_frame(), model, InferenceParams(tiling=tiling), use_tiles)
    assert r.tiles == [] and not r.too_small
    assert len(model.calls) == 1          # solo la pasada de frame completo
    assert (r.src == 0).all() and r.tiles_s == 0.0


def test_small_image_flags_too_small():
    model = FakeModel()
    r = detect_persons(np.zeros((360, 640, 3), dtype=np.uint8), model, InferenceParams())
    assert r.too_small and r.tiles == []
    assert len(model.calls) == 1


def test_empty_detections():
    def nothing(images, imgsz, conf):
        return [np.zeros((0, 6)) for _ in images]

    r = detect_persons(_frame(), nothing, InferenceParams())
    assert r.boxes.shape == (0, 6) and len(r.src) == 0


class _T:
    """Imita un tensor de torch: solo .cpu().numpy()."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)

    def __len__(self):
        return len(self.xyxy.a)


class _Results:
    def __init__(self, boxes):
        self.boxes = boxes


def test_results_to_boxes():
    r = _Results(_Boxes([[1, 2, 3, 4]], [0.5], [0]))
    out = results_to_boxes(r)
    assert out.shape == (1, 6) and out.dtype == np.float64
    np.testing.assert_allclose(out[0], [1, 2, 3, 4, 0.5, 0])
    assert results_to_boxes(_Results(None)).shape == (0, 6)
