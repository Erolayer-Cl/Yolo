"""Dibujos sobre la imagen: cajas de personas, HUD de texto, cruz central y avisos.

Lo usan person_detector (imagen anotada para Foxglove) y fpv_node (video FPV
para el navegador). Solo numpy + OpenCV: sin rclpy, testeable con pytest.

Cajas: ndarray (N, >=5) con columnas x1, y1, x2, y2, score[, class_id], en
píxeles de la imagen sobre la que se dibuja (mismo formato que tiling.py).
"""
import math

import cv2
import numpy as np

COLOR_FULL = (60, 220, 60)       # BGR: verde = vista por el frame completo (cerca)
COLOR_TILE = (0, 165, 255)       # naranja = vista por un tile (lejos)
COLOR_ALERT = (80, 80, 255)      # rojo claro: hay personas / aviso
WHITE = (255, 255, 255)


def enu_quat_to_heading_deg(qz: float, qw: float) -> float:
    """Yaw de un cuaternión ENU solo-yaw (como /drone/pose) -> rumbo 0..360,
    0 = Norte, horario."""
    yaw_enu = 2.0 * math.atan2(qz, qw)
    return math.degrees(math.pi / 2.0 - yaw_enu) % 360.0


def draw_boxes(img, boxes, src=None, scale: float = 1.0, ids=None):
    """Cajas con su score. `src[i]` verdadero = vino de un tile (naranja).
    `scale` convierte de píxeles de la detección a píxeles de `img`.
    `ids[i]` > 0 = id de persona del tracker, se antepone como '#id'."""
    if boxes is None or len(boxes) == 0:
        return img
    if src is None:
        src = np.zeros(len(boxes), dtype=bool)
    thick = max(2, img.shape[1] // 640)
    if ids is None:
        ids = np.zeros(len(boxes), dtype=int)
    for box, s, pid in zip(boxes, src, ids):
        x1, y1, x2, y2, score = (float(v) for v in box[:5])
        color = COLOR_TILE if s else COLOR_FULL
        p1 = (int(x1 * scale), int(y1 * scale))
        p2 = (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(img, p1, p2, color, thick)
        label = f'#{int(pid)} {score:.2f}' if pid else f'{score:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(img, label, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def draw_text_panel(img, lines, highlight_first: bool = False, width: int = 560):
    """Recuadro negro arriba a la izquierda con una línea de texto por renglón.
    La primera línea va en negrita, en rojo si `highlight_first`."""
    scale = max(0.5, img.shape[1] / 1600.0)
    lh = int(26 * scale)
    cv2.rectangle(img, (0, 0), (int(width * scale), lh * len(lines) + 10), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        color = COLOR_ALERT if (i == 0 and highlight_first) else WHITE
        cv2.putText(img, text, (8, lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, color, 1 + int(i == 0), cv2.LINE_AA)
    return img


def draw_crosshair(img, color=WHITE):
    """Cruz en el centro, con hueco al medio para no tapar lo que se apunta."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    size = max(12, w // 40)
    gap = size // 3
    thick = max(1, w // 960)
    for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(img, (cx + dx * gap, cy + dy * gap), (cx + dx * size, cy + dy * size),
                 color, thick, cv2.LINE_AA)
    return img


def draw_banner(img, text: str, color=COLOR_ALERT):
    """Aviso grande centrado (p.ej. 'SIN CAMARA')."""
    h, w = img.shape[:2]
    fs = max(0.8, w / 900.0)
    thick = max(2, int(fs * 2))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
    x, y = (w - tw) // 2, (h + th) // 2
    cv2.rectangle(img, (x - 12, y - th - 12), (x + tw + 12, y + 12), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, thick, cv2.LINE_AA)
    return img
