"""Cuándo una persona pasa a ser un waypoint del mapa, dónde se la ubica y cómo
se evitan duplicados. Todo en vivo, frame a frame.

tracker.py da un id a cada track confirmado (min_hits frames). Eso todavía no
es un waypoint:

  1. Espera: el track tiene que seguirse `min_seen_sec`, y se espera a tener una
     posición con error <= `max_error_m` (si no llega, hasta `max_wait_sec`; si
     el track se pierde antes, se anuncia con lo que haya). Un falso positivo
     que dura un instante no llega al mapa, y una persona lejana no se marca con
     ±20 m si acercándose se la puede ubicar mejor.
  2. Fusión: la posición de una persona es el promedio de TODAS sus
     observaciones, pesado por 1/error² (las cercanas y verticales pesan más).
     Una observación muy lejos de la fusión (más de `gate_sigma` sigmas) se
     descarta. El error de la fusión baja con cada observación, con un piso
     (`min_fused_ratio` del mejor error individual, y `min_error_m`): los
     errores de frames seguidos no son independientes (pitch, retraso), y sin
     piso el error informado sería irrealmente chico.
     Cada vez que la posición mejora o se mueve, evento 'fix' (como mucho cada
     `fix_period_sec`).
  3. Duplicados al anunciar: si cae cerca (`dedupe_radius_m`, o 2 sigma
     combinados si es mayor) de una persona ya marcada que NO se está viendo en
     este frame, es la misma que salió de cuadro y volvió: evento 'duplicate'.
     Aproximadas (sin posición propia): se comparan por la posición del DRON
     (`approx_radius_m`).
  4. Duplicados después: si al mejorar su posición dos personas marcadas quedan
     a menos de ese radio, y NUNCA se vieron en el mismo frame, son la misma:
     se fusionan en la más vieja y la otra se borra (evento 'merge').
     Dos personas vistas a la vez nunca se fusionan.
  Límite: dos personas distintas a menos de ese radio, vistas siempre en
  momentos distintos, quedan como una.

Los ids de persona son propios, consecutivos y no se reutilizan (ni tras un
reset ni tras un merge): en el mapa quedan las personas anteriores.

Python puro, testeable con pytest.
"""
import math
from dataclasses import dataclass, field, replace

from person_yolo.trail import haversine_m

INF = float('inf')
REJECT_RESET = 5            # descartes seguidos y coherentes -> la fusión estaba mal


@dataclass
class Observation:
    """Un track confirmado visto en este frame."""
    track_id: int
    lat: float = None           # de la persona en el suelo; None si no se pudo ubicar
    lon: float = None
    error_m: float = INF
    payload: dict = None        # datos del detector, se devuelven en el evento
    drone_lat: float = None     # posición del dron en este frame (None = sin GPS)
    drone_lon: float = None

    @property
    def located(self) -> bool:
        return self.lat is not None


@dataclass
class Event:
    kind: str                   # 'alert' (persona nueva) | 'fix' (posición mejor) |
                                # 'duplicate' (track nuevo que es una persona ya marcada) |
                                # 'merge' (person_id absorbió a `removed`, que se borra)
    person_id: int
    t: float                    # identidad de la persona: stamp de su alerta
    obs: Observation            # lat/lon/error_m = posición FUSIONADA de la persona
    removed: tuple = None       # merge: (id, t) de la persona que se borra


class _Fusion:
    """Promedio pesado por 1/error² de posiciones (lat/lon: en unos cientos de
    metros, promediar grados es lo mismo que promediar metros)."""

    def __init__(self):
        self.sw = self.slat = self.slon = 0.0
        self.best = INF
        self.n = 0
        self.rejected = []          # observaciones descartadas seguidas (ver PersonRegistry._feed)

    def add(self, lat, lon, err):
        w = 1.0 / max(err, 0.1) ** 2
        self.sw += w
        self.slat += w * lat
        self.slon += w * lon
        self.best = min(self.best, err)
        self.n += 1

    def absorb(self, other: '_Fusion'):
        self.sw += other.sw
        self.slat += other.slat
        self.slon += other.slon
        self.best = min(self.best, other.best)
        self.n += other.n

    @property
    def ok(self) -> bool:
        return self.n > 0

    def position(self):
        return self.slat / self.sw, self.slon / self.sw

    def error(self, min_ratio, min_error):
        return max(1.0 / math.sqrt(self.sw), min_ratio * self.best, min_error)


@dataclass
class _Pending:
    first: float
    last: float
    latest: Observation             # la última vista (para el payload del evento)
    fusion: _Fusion = field(default_factory=_Fusion)


@dataclass
class _Person:
    id: int
    t: float
    last: float
    fusion: _Fusion
    drone_lat: float = None         # dónde estaba el dron al marcarla
    drone_lon: float = None
    pub_lat: float = None           # última posición anunciada (alert/fix)
    pub_lon: float = None
    pub_err: float = INF
    pub_t: float = -INF

    @property
    def lat(self):
        return self.fusion.position()[0] if self.fusion.ok else None

    @property
    def lon(self):
        return self.fusion.position()[1] if self.fusion.ok else None


@dataclass
class PersonRegistry:
    min_seen_sec: float = 2.0
    max_wait_sec: float = 6.0
    forget_sec: float = 2.0         # track pendiente sin verse más que esto = perdido
    dedupe_radius_m: float = 4.0    # 0 = no buscar duplicados
    approx_radius_m: float = 10.0   # aproximadas: distancia del DRON; 0 = no buscar
    max_error_m: float = INF        # esperar (hasta max_wait_sec) una posición así de buena
    min_ratio: float = 0.8          # fix si el error baja a menos de esto veces el anunciado
    gate_sigma: float = 3.0         # observación a más de esto (en sigmas) = descartada
    min_fused_ratio: float = 0.5    # piso del error fusionado, relativo al mejor individual
    min_error_m: float = 1.0        # piso absoluto del error fusionado
    fix_period_sec: float = 1.0     # como mucho un fix por persona cada esto
    persons: dict = field(default_factory=dict)         # id -> _Person
    pending: dict = field(default_factory=dict)         # track_id -> _Pending
    track_person: dict = field(default_factory=dict)    # track_id -> id de persona
    together: set = field(default_factory=set)          # pares de ids vistos en un mismo frame
    total: int = 0
    _next_id: int = 1

    def reset(self):
        """Total a 0 y se olvidan las personas (no se buscan duplicados contra
        ellas). Los ids siguen desde donde iban."""
        self.persons, self.pending, self.track_person = {}, {}, {}
        self.together = set()
        self.total = 0

    def person_id(self, track_id) -> int:
        """Id de persona de un track; 0 si todavía no es un waypoint."""
        return self.track_person.get(int(track_id), 0)

    # -------------------------------------------------------------- update
    def update(self, now: float, stamp_t: float, observations) -> list:
        """`now`: reloj monotónico (s). `stamp_t`: stamp del frame, identidad de
        las personas que se anuncien ahora. Devuelve la lista de eventos."""
        events = []
        seen_now = set()
        touched = []                        # personas cuya posición cambió en este frame
        for o in observations:
            pid = self.track_person.get(o.track_id)
            if pid is not None:
                p = self.persons.get(pid)
                if p is None:               # olvidada por un reset
                    continue
                p.last = now
                seen_now.add(pid)
                if o.located and self._feed(p.fusion, o):
                    touched.append((p, o))
                continue
            pend = self.pending.get(o.track_id)
            if pend is None:
                pend = self.pending[o.track_id] = _Pending(now, now, o)
            pend.last, pend.latest = now, o
            if o.located:
                self._feed(pend.fusion, o)

        for tid, pend in list(self.pending.items()):
            seen = pend.last - pend.first
            lost = now - pend.last > self.forget_sec
            good = pend.fusion.ok and self._err(pend.fusion) <= self.max_error_m
            if seen >= self.min_seen_sec and (good or lost or seen >= self.max_wait_sec):
                del self.pending[tid]
                events.extend(self._announce(tid, pend, now, stamp_t, seen_now))
            elif lost:
                del self.pending[tid]       # no duró lo suficiente: falso positivo

        # Vistas en el mismo frame = personas distintas, para siempre.
        ids = sorted(seen_now)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                self.together.add((a, b))

        for p, o in touched:
            if p.id not in self.persons:    # ya absorbida por un merge en este frame
                continue
            other = self._mergeable(p)
            if other is not None:
                events.append(self._merge(p, other, o, now))
            elif self._worth_fix(p, now):
                events.append(self._event('fix', p, o, now))
        return events

    # ------------------------------------------------------------- fusión
    def _err(self, f: _Fusion) -> float:
        return f.error(self.min_fused_ratio, self.min_error_m)

    def _near(self, f: _Fusion, o: Observation) -> bool:
        lat, lon = f.position()
        d = haversine_m(o.lat, o.lon, lat, lon)
        return d <= self.gate_sigma * math.hypot(o.error_m, self._err(f)) + self.min_error_m

    def _feed(self, f: _Fusion, o: Observation) -> bool:
        """Suma `o` a la fusión. Con 2+ observaciones descarta la que cae muy lejos
        (un rayo rasante que salió mal). Pero si se descartan `REJECT_RESET`
        seguidas y son coherentes entre sí, la que está mal es la fusión (las
        primeras vistas, lejos, con un error informado demasiado chico): se
        reinicia con ellas. True si la fusión cambió."""
        if f.n < 2 or self._near(f, o):
            f.rejected = []
            f.add(o.lat, o.lon, o.error_m)
            return True
        f.rejected.append(o)
        if len(f.rejected) < REJECT_RESET:
            return False
        alt = _Fusion()
        for r in f.rejected:
            alt.add(r.lat, r.lon, r.error_m)
        if not all(self._near(alt, r) for r in f.rejected):
            f.rejected = f.rejected[1:]     # ruido, no una tendencia: seguir esperando
            return False
        f.sw, f.slat, f.slon, f.best, f.n = alt.sw, alt.slat, alt.slon, alt.best, alt.n
        f.rejected = []
        return True

    def _worth_fix(self, p: _Person, now) -> bool:
        if p.pub_lat is None:               # aproximada que ahora tiene posición: ya
            return True
        if now - p.pub_t < self.fix_period_sec:
            return False
        err = self._err(p.fusion)
        moved = haversine_m(p.lat, p.lon, p.pub_lat, p.pub_lon)
        return err < p.pub_err * self.min_ratio or moved > max(2.0, 0.5 * err)

    def _event(self, kind, p: _Person, o: Observation, now, removed=None) -> Event:
        """Evento con la posición fusionada de `p` (o la observación tal cual si
        p no tiene posición)."""
        if p.fusion.ok:
            err = self._err(p.fusion)
            o = replace(o, lat=p.lat, lon=p.lon, error_m=err)
            p.pub_lat, p.pub_lon, p.pub_err = p.lat, p.lon, err
        p.pub_t = now
        return Event(kind, p.id, p.t, o, removed)

    # ---------------------------------------------------------- anunciar
    def _announce(self, tid, pend: _Pending, now, stamp_t, seen_now) -> list:
        o = pend.latest
        located = pend.fusion.ok
        if located:
            lat, lon = pend.fusion.position()
            o = replace(o, lat=lat, lon=lon, error_m=self._err(pend.fusion))
        dup = (self._nearest_unseen(o, seen_now, pend.fusion.best) if located
               else self._nearest_unseen_approx(o, seen_now))
        if dup is not None:
            self.track_person[tid] = dup.id
            dup.last = now
            seen_now.add(dup.id)
            out = [Event('duplicate', dup.id, dup.t, o)]
            if located:
                dup.fusion.absorb(pend.fusion)
                if self._worth_fix(dup, now):
                    out.append(self._event('fix', dup, o, now))
            return out
        p = _Person(self._next_id, stamp_t, now, pend.fusion, o.drone_lat, o.drone_lon)
        self._next_id += 1
        self.total += 1
        self.persons[p.id] = p
        self.track_person[tid] = p.id
        seen_now.add(p.id)
        return [self._event('alert', p, o, now)]

    def _limit(self, err_a, err_b) -> float:
        """Radio para considerar dos posiciones la misma persona. Se usa el error
        de UNA observación (el mejor), no el fusionado: el error sistemático
        (pitch, retraso, terreno) no baja promediando, y con el fusionado, que es
        más chico, dos marcas de la misma persona quedarían separadas."""
        return max(self.dedupe_radius_m, 2.0 * math.hypot(err_a, err_b))

    def _nearest_unseen(self, o: Observation, seen_now, err_one):
        """`err_one`: mejor error de UNA observación de `o` (ver _limit)."""
        if self.dedupe_radius_m <= 0:
            return None
        best, best_d = None, INF
        for p in self.persons.values():
            if p.id in seen_now or not p.fusion.ok:
                continue
            d = haversine_m(o.lat, o.lon, p.lat, p.lon)
            if d <= self._limit(err_one, p.fusion.best) and d < best_d:
                best, best_d = p, d
        return best

    def _nearest_unseen_approx(self, o: Observation, seen_now):
        """Aproximada ya marcada, no vista ahora, marcada con el dron cerca de
        donde está ahora. Solo contra aproximadas: una ubicada tiene su propia
        posición, que puede estar lejos del dron."""
        if self.approx_radius_m <= 0 or o.drone_lat is None:
            return None
        best, best_d = None, INF
        for p in self.persons.values():
            if p.id in seen_now or p.fusion.ok or p.drone_lat is None:
                continue
            d = haversine_m(o.drone_lat, o.drone_lon, p.drone_lat, p.drone_lon)
            if d <= self.approx_radius_m and d < best_d:
                best, best_d = p, d
        return best

    # ------------------------------------------------------------- merge
    def _mergeable(self, p: _Person):
        """Otra persona ubicada, cerca de p, que nunca se vio junto con p."""
        if self.dedupe_radius_m <= 0 or not p.fusion.ok:
            return None
        best, best_d = None, INF
        for q in self.persons.values():
            if q.id == p.id or not q.fusion.ok:
                continue
            if (min(p.id, q.id), max(p.id, q.id)) in self.together:
                continue
            d = haversine_m(p.lat, p.lon, q.lat, q.lon)
            if d <= self._limit(p.fusion.best, q.fusion.best) and d < best_d:
                best, best_d = q, d
        return best

    def _merge(self, a: _Person, b: _Person, o: Observation, now) -> Event:
        """Fusiona las dos en la más vieja (id menor); la otra se borra."""
        keep, drop = (a, b) if a.id < b.id else (b, a)
        keep.fusion.absorb(drop.fusion)
        keep.last = max(keep.last, drop.last)
        for tid, pid in self.track_person.items():
            if pid == drop.id:
                self.track_person[tid] = keep.id
        # Lo que no se podía fusionar con `drop` tampoco con `keep`.
        for x, y in list(self.together):
            if drop.id in (x, y):
                other = y if x == drop.id else x
                if other != keep.id:
                    self.together.add((min(other, keep.id), max(other, keep.id)))
        del self.persons[drop.id]
        self.total -= 1
        return self._event('merge', keep, o, now, removed=(drop.id, drop.t))
