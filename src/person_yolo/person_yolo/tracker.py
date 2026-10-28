"""Conteo de personas únicas: tracker liviano sobre las cajas ya fusionadas.

No se usa `model.track()` de ultralytics porque las cajas finales salen de
fusionar frame completo + tiles (tiling.py), y el tracker de ultralytics solo ve
una de las dos pasadas. Acá se trackea el resultado final.

Por frame:
  1. Se predice dónde está cada track (velocidad constante del centro, en px/s:
     el dron se mueve y las personas se desplazan en la imagen).
  2. Asociación greedy caja <-> track por distancia de centros normalizada por el
     tamaño de la caja (anda con cajas chicas de personas lejanas, donde el IoU
     cae a 0 con un par de píxeles de corrimiento) o por IoU.
  3. Un track recién CONFIRMADO (min_hits frames) suma 1 al total. Así un falso
     positivo de un solo frame no cuenta, y la misma persona no se cuenta dos
     veces mientras se la siga viendo.
  4. Un track se borra si lleva más de max_age_sec sin detección Y además faltó
     en max_missed inferencias seguidas. Tolera los frames sin tiles
     (tiling.every_n_frames) y oclusiones cortas; y si la inferencia va lenta
     (CPU, Jetson caliente) no se olvida a quien se sigue viendo en cada frame
     (si no, cada persona se volvería a contar en cada inferencia lenta).

Limitación: si una persona sale de cámara y vuelve a entrar después de
max_age_sec, se cuenta de nuevo (no hay re-identificación por apariencia).

Python puro + numpy (sin rclpy), testeable con pytest.
"""
from dataclasses import dataclass, field

import numpy as np


def _center(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    box: np.ndarray                 # x1, y1, x2, y2 de la última detección
    t_last: float
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))   # px/s del centro
    hits: int = 1
    missed: int = 0                 # inferencias seguidas sin detección
    id: int = 0                     # 0 = todavía sin confirmar

    def predicted(self, t):
        """Caja desplazada según la velocidad hasta el instante `t`."""
        dx, dy = self.vel * max(0.0, t - self.t_last)
        return self.box + np.array([dx, dy, dx, dy])


@dataclass
class PersonCounter:
    min_hits: int = 3               # frames con detección para contar a alguien
    max_age_sec: float = 1.5        # sin verlo más que esto...
    max_missed: int = 3             # ...y en tantas inferencias seguidas -> se olvida
    iou_thr: float = 0.2            # asocia si IoU >= esto...
    dist_thr: float = 1.0           # ...o si dist. de centros <= esto * diagonal
    min_size_px: float = 16.0       # piso de la diagonal (cajas diminutas)
    tracks: list = field(default_factory=list)
    total: int = 0                  # personas distintas confirmadas
    new_ids: list = field(default_factory=list)     # confirmadas en el último update
    _next_id: int = 1

    def reset(self):
        self.tracks = []
        self.total = 0
        self.new_ids = []
        self._next_id = 1

    def update(self, boxes, t: float) -> np.ndarray:
        """Asocia las cajas (N, >=4) del instante `t` (segundos) a tracks.

        Devuelve un array (N,) con el id de persona de cada caja, 0 si el track
        todavía no está confirmado. Los ids que se confirmaron en esta llamada
        (personas nuevas) quedan en `new_ids`.
        """
        self.new_ids = []
        boxes = np.asarray(boxes, dtype=np.float64)
        if boxes.size == 0:
            boxes = np.zeros((0, 4))
        self.tracks = [tr for tr in self.tracks
                       if t - tr.t_last <= self.max_age_sec or tr.missed < self.max_missed]

        pairs = []
        for ti, tr in enumerate(self.tracks):
            pred = tr.predicted(t)
            c_pred = _center(pred)
            scale = max(self.min_size_px, float(np.hypot(pred[2] - pred[0],
                                                         pred[3] - pred[1])))
            for di in range(len(boxes)):
                b = boxes[di, :4]
                dist = float(np.linalg.norm(_center(b) - c_pred)) / scale
                if dist <= self.dist_thr or _iou(pred, b) >= self.iou_thr:
                    pairs.append((dist, ti, di))
        pairs.sort()

        ids = np.zeros(len(boxes), dtype=np.int64)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self._update_track(self.tracks[ti], boxes[di, :4], t)
            ids[di] = self.tracks[ti].id
        for ti, tr in enumerate(self.tracks):
            tr.missed = 0 if ti in used_t else tr.missed + 1

        for di in range(len(boxes)):
            if di not in used_d:
                tr = Track(box=boxes[di, :4].copy(), t_last=t)
                self.tracks.append(tr)
                self._confirm(tr)
                ids[di] = tr.id
        return ids

    def _update_track(self, tr: Track, box, t):
        dt = t - tr.t_last
        if dt > 1e-6:
            v = (_center(box) - _center(tr.box)) / dt
            tr.vel = 0.5 * tr.vel + 0.5 * v         # suaviza el jitter de las cajas
        tr.box = box.copy()
        tr.t_last = t
        tr.hits += 1
        self._confirm(tr)

    def _confirm(self, tr: Track):
        if tr.id == 0 and tr.hits >= self.min_hits:
            tr.id = self._next_id
            self._next_id += 1
            self.total += 1
            self.new_ids.append(tr.id)

    @property
    def active(self) -> int:
        """Personas confirmadas que se están siguiendo ahora."""
        return sum(1 for tr in self.tracks if tr.id)
