import numpy as np
import pytest

from person_yolo.tiling import make_tiles, merge, nms, remap_tile_boxes


def _covered(w, h, tiles):
    mask = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in tiles:
        mask[y0:y1, x0:x1] = True
    return mask.all()


@pytest.mark.parametrize('w,h', [(1920, 1080), (1280, 720), (640, 360), (2000, 641)])
def test_tiles_cover_image_and_fit(w, h):
    tiles = make_tiles(w, h, 640, 0.2)
    assert _covered(w, h, tiles)
    for x0, y0, x1, y1 in tiles:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        assert x1 - x0 == min(640, w) and y1 - y0 == min(640, h)


def test_small_image_is_single_tile():
    assert make_tiles(640, 360, 640, 0.2) == [(0, 0, 640, 360)]


def test_1080p_tile_count():
    # step = 512 -> x: 0, 512, 1024, 1280 ; y: 0, 440
    tiles = make_tiles(1920, 1080, 640, 0.2)
    assert len(tiles) == 8
    assert (1280, 440, 1920, 1080) in tiles


def test_invalid_args():
    with pytest.raises(ValueError):
        make_tiles(100, 100, 0, 0.2)
    with pytest.raises(ValueError):
        make_tiles(100, 100, 64, 1.0)


def test_remap_offsets_and_drops_truncated():
    rect = (512, 440, 1152, 1080)       # tile interior en x, toca abajo en y
    boxes = np.array([
        [100, 100, 120, 140, 0.9, 0],   # adentro -> se queda, desplazada
        [0, 100, 20, 140, 0.9, 0],      # toca borde izquierdo interno -> fuera
        [100, 600, 120, 640, 0.9, 0],   # toca borde inferior = borde de imagen -> queda
    ], dtype=float)
    out = remap_tile_boxes(boxes, rect, 1920, 1080)
    assert len(out) == 2
    np.testing.assert_allclose(out[0, :4], [612, 540, 632, 580])
    np.testing.assert_allclose(out[1, :4], [612, 1040, 632, 1080])


def test_nms_keeps_best_and_separate_classes():
    boxes = np.array([
        [0, 0, 10, 10, 0.5, 0],
        [1, 1, 11, 11, 0.9, 0],
        [1, 1, 11, 11, 0.8, 1],
        [50, 50, 60, 60, 0.4, 0],
    ], dtype=float)
    keep = nms(boxes, 0.5)
    assert sorted(keep) == [1, 2, 3]
    assert keep[0] == 1


def test_ios_removes_contained_box():
    boxes = np.array([[0, 0, 100, 200, 0.9, 0], [0, 0, 50, 100, 0.6, 0]], dtype=float)
    assert len(nms(boxes, 0.5, 'iou')) == 2
    assert nms(boxes, 0.5, 'ios') == [0]


def test_merge_dedups_overlap_between_tiles_and_full():
    w, h = 1920, 1080
    person_global = [600, 300, 610, 320]          # persona lejana chica
    t1 = (512, 0, 1152, 640)
    t2 = (0, 0, 640, 640)
    in_t1 = [600 - 512, 300, 610 - 512, 320, 0.7, 0]
    in_t2 = [600, 300, 610, 320, 0.6, 0]          # aparece en el solape de ambos
    full = np.array([[100, 100, 300, 700, 0.95, 0]])   # persona cercana grande
    boxes, src = merge(full, [(t1, [in_t1]), (t2, [in_t2])], w, h)
    assert len(boxes) == 2
    assert list(src) == [0, 1]
    np.testing.assert_allclose(boxes[1, :4], person_global)
    assert boxes[1, 4] == pytest.approx(0.7)


def test_merge_empty():
    boxes, src = merge(np.zeros((0, 6)), [((0, 0, 640, 640), [])], 1280, 720)
    assert boxes.shape == (0, 6) and len(src) == 0
