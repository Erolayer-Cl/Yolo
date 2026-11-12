"""Historial corto de un estado del dron (actitud, GPS, pose) por timestamp.

Para ubicar a una persona hay que usar la actitud y la posición del dron en el
instante en que se SACÓ la imagen, no las de ahora. En SITL la imagen llega con
0.3-0.7 s de atraso (render + RPC de AirSim + inferencia): si el dron gira o se
inclina en ese tiempo, el rayo de la cámara apunta a otro lado y la persona cae
5-15 m corrida. person_detector guarda acá cada mensaje con su header.stamp y
busca el más cercano al stamp de la imagen.

Python puro, testeable con pytest.
"""
from bisect import bisect_left
from collections import deque


class History:
    """Muestras (t, valor) en orden de llegada; se guardan `keep_sec` segundos."""

    def __init__(self, keep_sec: float = 3.0):
        self.keep_sec = keep_sec
        self._t = deque()
        self._v = deque()

    def __len__(self):
        return len(self._t)

    def add(self, t: float, value):
        if self._t and t < self._t[-1]:      # stamp para atrás (rosbag en loop): empezar de nuevo
            self._t.clear()
            self._v.clear()
        self._t.append(t)
        self._v.append(value)
        while self._t and self._t[0] < t - self.keep_sec:
            self._t.popleft()
            self._v.popleft()

    def at(self, t: float, max_gap: float = 0.25):
        """(valor, t_muestra) de la muestra más cercana a `t`, o None si no hay
        ninguna a menos de `max_gap` segundos (sin historial, o `t` fuera de él)."""
        if not self._t:
            return None
        i = bisect_left(self._t, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(self._t)]
        j = min(cands, key=lambda k: abs(self._t[k] - t))
        if abs(self._t[j] - t) > max_gap:
            return None
        return self._v[j], self._t[j]
