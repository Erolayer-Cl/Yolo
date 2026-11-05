"""Cuándo una persona pasa a ser un waypoint del mapa, y cómo se evitan duplicados.

tracker.py da un id a cada track confirmado (min_hits frames). Eso todavía no
es un waypoint:

  1. Espera: el track tiene que seguirse `min_seen_sec`. Un falso positivo que
     dura un instante no llega al mapa, y mientras tanto se junta la mejor
     posición (geolocate.py), para que el waypoint salga ya ubicado. Si a los
     `min_seen_sec` todavía no hay posición se espera hasta `max_wait_sec`; si el
     track se pierde antes, se anuncia con lo que haya.
  2. Duplicados: si al anunciarse cae cerca (`dedupe_radius_m`, o 2 sigma
     combinados si es mayor) de una persona ya marcada que NO se está viendo en
     este frame, es la misma que salió de cuadro y volvió: se le asigna ese id
     en vez de crear otro waypoint. Dos personas vistas a la vez nunca se
     fusionan. Límite: dos personas distintas a menos de ese radio, vistas en
     momentos distintos, quedan como una.
  3. Mejoras: cada vez que el error de una persona baja por debajo de
     `min_ratio` veces el mejor, evento 'fix'.

Los ids de persona son propios, consecutivos y no se reutilizan (ni tras un
reset): en el mapa quedan las personas anteriores.

Python puro, testeable con pytest.
"""
import math
from dataclasses import dataclass, field

from person_yolo.trail import haversine_m

INF = float('inf')


@dataclass
class Observation:
    """Un track confirmado visto en este frame."""
    track_id: int
    lat: float = None           # de la persona en el suelo; None si no se pudo ubicar
    lon: float = None
    error_m: float = INF
    payload: dict = None        # datos del detector, se devuelven en el evento

    @property
    def located(self) -> bool:
        return self.lat is not None


@dataclass
class Event:
    kind: str                   # 'alert' (persona nueva) | 'fix' (mejor posición) |
                                # 'duplicate' (track nuevo que es una persona ya marcada)
    person_id: int
    t: float                    # identidad de la persona: stamp de su alerta
    obs: Observation


@dataclass
class _Pending:
    first: float
    last: float
    best: Observation


@dataclass
class _Person:
    id: int
    t: float
    lat: float
    lon: float
    error_m: float
    last: float


@dataclass
class PersonRegistry:
    min_seen_sec: float = 2.0
    max_wait_sec: float = 6.0
    forget_sec: float = 2.0         # track pendiente sin verse más que esto = perdido
    dedupe_radius_m: float = 4.0    # 0 = no buscar duplicados
    min_ratio: float = 0.8
    persons: dict = field(default_factory=dict)         # id -> _Person
    pending: dict = field(default_factory=dict)         # track_id -> _Pending
    track_person: dict = field(default_factory=dict)    # track_id -> id de persona
    total: int = 0
    _next_id: int = 1

    def reset(self):
        """Total a 0 y se olvidan las personas (no se buscan duplicados contra
        ellas). Los ids siguen desde donde iban."""
        self.persons, self.pending, self.track_person = {}, {}, {}
        self.total = 0

    def person_id(self, track_id) -> int:
        """Id de persona de un track; 0 si todavía no es un waypoint."""
        return self.track_person.get(int(track_id), 0)

    def update(self, now: float, stamp_t: float, observations) -> list:
        """`now`: reloj monotónico (s). `stamp_t`: stamp del frame, identidad de
        las personas que se anuncien ahora. Devuelve la lista de eventos."""
        events = []
        seen_now = set()
        for o in observations:
            pid = self.track_person.get(o.track_id)
            if pid is not None:
                p = self.persons.get(pid)
                if p is None:               # olvidada por un reset
                    continue
                p.last = now
                seen_now.add(pid)
                if self._improves(p, o):
                    events.append(self._fix(p, o))
                continue
            pend = self.pending.get(o.track_id)
            if pend is None:
                self.pending[o.track_id] = _Pending(now, now, o)
                continue
            pend.last = now
            if o.located and o.error_m < pend.best.error_m:
                pend.best = o
            elif not pend.best.located:
                pend.best = o               # sin posición: la más reciente

        for tid, pend in list(self.pending.items()):
            seen = pend.last - pend.first
            lost = now - pend.last > self.forget_sec
            if seen >= self.min_seen_sec and (pend.best.located or lost
                                              or seen >= self.max_wait_sec):
                del self.pending[tid]
                events.extend(self._announce(tid, pend.best, now, stamp_t, seen_now))
            elif lost:
                del self.pending[tid]       # no duró lo suficiente: falso positivo
        return events

    def _improves(self, p: _Person, o: Observation) -> bool:
        return o.located and o.error_m < p.error_m * self.min_ratio

    def _fix(self, p: _Person, o: Observation) -> Event:
        p.lat, p.lon, p.error_m = o.lat, o.lon, o.error_m
        return Event('fix', p.id, p.t, o)

    def _announce(self, tid, o: Observation, now, stamp_t, seen_now) -> list:
        dup = self._nearest_unseen(o, seen_now) if o.located else None
        if dup is not None:
            self.track_person[tid] = dup.id
            dup.last = now
            seen_now.add(dup.id)
            out = [Event('duplicate', dup.id, dup.t, o)]
            if self._improves(dup, o):
                out.append(self._fix(dup, o))
            return out
        p = _Person(self._next_id, stamp_t, o.lat, o.lon,
                    o.error_m if o.located else INF, now)
        self._next_id += 1
        self.total += 1
        self.persons[p.id] = p
        self.track_person[tid] = p.id
        seen_now.add(p.id)
        return [Event('alert', p.id, p.t, o)]

    def _nearest_unseen(self, o: Observation, seen_now):
        if self.dedupe_radius_m <= 0:
            return None
        best, best_d = None, INF
        for p in self.persons.values():
            if p.id in seen_now or p.lat is None:
                continue
            d = haversine_m(o.lat, o.lon, p.lat, p.lon)
            limit = max(self.dedupe_radius_m, 2.0 * math.hypot(o.error_m, p.error_m))
            if d <= limit and d < best_d:
                best, best_d = p, d
        return best
