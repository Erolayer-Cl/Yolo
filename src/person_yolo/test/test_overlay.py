import numpy as np
import pytest

from person_yolo.overlay import (
    COLOR_FULL,
    COLOR_TILE,
    draw_banner,
    draw_boxes,
    draw_crosshair,
    draw_text_panel,
    enu_quat_to_heading_deg,
)


def _yaw_quat(yaw_enu):
    return np.sin(yaw_enu / 2.0), np.cos(yaw_enu / 2.0)


@pytest.mark.parametrize('yaw_enu,heading', [
    (np.pi / 2, 0.0),      # ENU mirando al Norte -> rumbo 0
    (0.0, 90.0),           # Este -> 90
    (-np.pi / 2, 180.0),   # Sur -> 180
    (np.pi, 270.0),        # Oeste -> 270
])
def test_heading(yaw_enu, heading):
    assert enu_quat_to_heading_deg(*_yaw_quat(yaw_enu)) == pytest.approx(heading, abs=1e-6)


def _img(w=640, h=360):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_boxes_colors_and_scale():
    img = _img()
    boxes = np.array([[100, 100, 200, 200, 0.9, 0], [300, 100, 400, 200, 0.5, 0]])
    draw_boxes(img, boxes, src=np.array([False, True]))
    assert tuple(img[150, 100]) == COLOR_FULL          # borde izquierdo de la 1ra
    assert tuple(img[150, 300]) == COLOR_TILE          # borde izquierdo de la 2da

    half = _img(320, 180)
    draw_boxes(half, boxes[:, :5], scale=0.5)          # 5 columnas, sin src
    assert tuple(half[75, 50]) == COLOR_FULL           # (100,150) * 0.5


def test_empty_boxes_leave_image_untouched():
    img = _img()
    draw_boxes(img, np.zeros((0, 5)))
    draw_boxes(img, None)
    assert not img.any()


def test_hud_crosshair_banner_draw_something():
    for fn in (lambda i: draw_text_panel(i, ['A', 'B'], highlight_first=True),
               draw_crosshair,
               lambda i: draw_banner(i, 'SIN CAMARA')):
        img = _img()
        out = fn(img)
        assert out is img and img.any()


def test_crosshair_leaves_center_free():
    img = _img()
    draw_crosshair(img)
    assert not img[180, 320].any()
