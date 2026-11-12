"""Recorrido del dron y personas encontradas (una por alerta, en su posición
en el suelo si se pudo ubicar, ver geolocate.py).

Python puro (sin rclpy): decimación por distancia y export a GeoJSON (panel
Map de Foxglove), KML (Google Earth) y CSV. Lo usa `trail_node`.
"""
import csv
import io
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from xml.sax.saxutils import escape

EARTH_R = 6371008.8
RING_SEGMENTS = 36          # lados del anillo de la zona de búsqueda
MAX_GRID_LINES = 41         # por eje; si no alcanza, se agranda el paso


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(min(1.0, a)))


def _north_east_m(origin, lat, lon):
    """Metros al norte y al este de `origin` (aproximación local, ver geolocate.py)."""
    north = math.radians(lat - origin.lat) * EARTH_R
    east = math.radians(lon - origin.lon) * EARTH_R * math.cos(math.radians(origin.lat))
    return north, east


def offset_to_latlon(lat, lon, north_m, east_m):
    lat2 = lat + math.degrees(north_m / EARTH_R)
    lon2 = lon + math.degrees(east_m / (EARTH_R * math.cos(math.radians(lat))))
    return lat2, lon2


def _offset_lon_lat(origin, north_m, east_m):
    """[lon, lat] a esa distancia de `origin`, en el orden de GeoJSON."""
    lat, lon = offset_to_latlon(origin.lat, origin.lon, north_m, east_m)
    return [lon, lat]


@dataclass
class TrailPoint:
    t: float                # segundos (stamp ROS)
    lat: float
    lon: float
    alt_msl: float
    alt_rel: float          # altura sobre home; NaN si no hay pose local
    heading_deg: float      # 0 = Norte, horario; NaN si no hay


@dataclass
class PersonSpot:
    """Una persona distinta (una alerta de person_detector)."""
    t: float
    id: int
    lat: float              # de la PERSONA si se pudo ubicar; si no, del dron
    lon: float
    located: bool           # True = posición en el suelo; False = la del dron
    error_m: float          # 1 sigma; NaN si no se ubicó
    score: float
    drone_lat: float
    drone_lon: float
    # Marca a mano desde la web (Trail.add_mark): la posición es la del DRON en ese
    # momento. id negativo (-1, -2...) para no chocar con los del detector;
    # se muestra como M1, M2...
    manual: bool = False

    @property
    def label(self) -> str:
        return f'M{-self.id}' if self.manual else str(self.id)

    def has_error(self) -> bool:
        return self.located and not math.isnan(self.error_m)

    def coords_text(self) -> str:
        """Coordenadas para rescatarla, o aviso si solo se sabe dónde estaba el dron."""
        if self.manual:
            return f'{self.lat:.6f}, {self.lon:.6f} (marca manual, posición del dron)'
        if not self.located:
            return 'APROXIMADA, posición del dron'
        err = f' ±{self.error_m:.1f} m' if self.has_error() else ''
        return f'{self.lat:.6f}, {self.lon:.6f}{err}'


def _valid_latlon(lat, lon):
    return (math.isfinite(lat) and math.isfinite(lon)
            and abs(lat) <= 90.0 and abs(lon) <= 180.0)


def spot_from_alert(al):
    """PersonSpot de un JSON de /yolo/person_alert o /yolo/person_fix. None si no
    trae una posición del dron válida: un dato malo no puede tirar el nodo ni
    colarse como NaN en el GeoJSON del mapa."""
    try:
        drone_lat, drone_lon = float(al['lat']), float(al['lon'])
        t = float(al.get('t', 0.0))
        pid = int(al.get('id', 0))
        score = float(al.get('score', 0.0))
        err = al.get('person_error_m')
        err = float('nan') if err is None else float(err)
        plat, plon = al.get('person_lat'), al.get('person_lon')
        located = plat is not None and plon is not None
        if located:
            plat, plon = float(plat), float(plon)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if not (_valid_latlon(drone_lat, drone_lon) and math.isfinite(t)):
        return None
    located = located and _valid_latlon(plat, plon)
    return PersonSpot(t=t, id=pid, lat=plat if located else drone_lat,
                      lon=plon if located else drone_lon, located=located,
                      error_m=err if located else float('nan'),
                      score=score, drone_lat=drone_lat, drone_lon=drone_lon)


@dataclass
class Trail:
    """Recorrido decimado: un punto nuevo solo si el dron se movió.

    Con el dron quieto en hover el GPS sigue llegando a 5-10 Hz; sin decimar
    la línea crecería sin parar y el mensaje al mapa se volvería enorme.
    """
    min_step_m: float = 1.0
    max_points: int = 5000
    points: list = field(default_factory=list)
    persons: list = field(default_factory=list)
    home: TrailPoint = None
    total_m: float = 0.0
    dirty: bool = False
    marks_n: int = 0                # marcas manuales creadas (no se reutilizan)
    # (id, t) de personas borradas por un merge: un fix que llega tarde (ya estaba
    # en la cola de ROS) no tiene que volver a crearlas.
    removed: set = field(default_factory=set)

    def add(self, p: TrailPoint) -> bool:
        """Agrega `p` si se alejó >= min_step_m (3D) del último. True si entró."""
        if self.home is None:
            self.home = p
        if self.points:
            last = self.points[-1]
            dh = haversine_m(last.lat, last.lon, p.lat, p.lon)
            dz = 0.0
            if not (math.isnan(p.alt_rel) or math.isnan(last.alt_rel)):
                dz = p.alt_rel - last.alt_rel
            d = math.hypot(dh, dz)
            if d < self.min_step_m:
                return False
            self.total_m += dh
        self.points.append(p)
        if len(self.points) > self.max_points:
            self._thin()
        self.dirty = True
        return True

    def _thin(self):
        # Se descarta uno de cada dos (salvo el primero y el último) en vez de
        # los más viejos: así se sigue viendo el recorrido COMPLETO, solo que
        # con menos resolución, que es justo lo que importa en el mapa.
        pts = self.points
        self.points = pts[:-1:2] + [pts[-1]]

    def add_person(self, spot: PersonSpot):
        """Registra una persona distinta, o le mejora la posición si ya está
        (mismo id y misma `t`, la de su primera alerta: si el detector se reinicia,
        un id repetido es otra persona). No se fusiona nada más: dos personas
        juntas tienen que quedar como dos puntos. Una persona ya borrada por un
        merge no se vuelve a crear."""
        if (spot.id, round(spot.t, 3)) in self.removed:
            return
        for i, p in enumerate(self.persons):
            if p.id == spot.id and abs(p.t - spot.t) < 1e-3:
                if spot.located or not p.located:
                    self.persons[i] = spot
                    self.dirty = True
                return
        self.persons.append(spot)
        self.dirty = True

    def remove_person(self, pid: int, t: float) -> bool:
        """Borra la persona (id, t): el detector se dio cuenta de que era otra ya
        marcada (/yolo/person_merge). True si estaba. Queda recordada: no se
        vuelve a crear aunque llegue después un fix suyo."""
        self.removed.add((pid, round(t, 3)))
        n = len(self.persons)
        self.persons = [p for p in self.persons
                        if p.manual or not (p.id == pid and abs(p.t - t) < 1e-3)]
        self.dirty = self.dirty or len(self.persons) != n
        return len(self.persons) != n

    def add_mark(self, t: float, lat: float, lon: float) -> PersonSpot:
        """Marca a mano en la posición del dron (botón de la web)."""
        self.marks_n += 1
        spot = PersonSpot(t=t, id=-self.marks_n, lat=lat, lon=lon, located=True,
                          error_m=float('nan'), score=0.0, drone_lat=lat, drone_lon=lon,
                          manual=True)
        self.persons.append(spot)
        self.dirty = True
        return spot

    def remove_mark(self, pid: int) -> bool:
        """Borra una marca manual (id negativo). Las del detector no se borran a mano."""
        n = len(self.persons)
        self.persons = [p for p in self.persons if not (p.manual and p.id == pid)]
        self.dirty = self.dirty or len(self.persons) != n
        return len(self.persons) != n

    # ------------------------------------------------------------- exports
    def to_geojson(self) -> dict:
        feats = []
        if len(self.points) >= 2:
            feats.append({
                'type': 'Feature',
                'geometry': {'type': 'LineString',
                             'coordinates': [[p.lon, p.lat] for p in self.points]},
                'properties': {
                    'name': 'recorrido',
                    'length_m': round(self.total_m, 1),
                    'style': {'color': '#1e90ff', 'weight': 4, 'opacity': 0.9},
                },
            })
        if self.home is not None:
            feats.append(_point(self.home.lon, self.home.lat, {
                'name': 'home',
                'style': {'color': '#2ecc71', 'radius': 8, 'fillOpacity': 0.9},
            }))
        # Los anillos van antes que los puntos: así quedan dibujados debajo.
        feats.extend(_search_ring(p) for p in self.persons if p.has_error())
        for p in self.persons:
            # Rojo = ubicada en el suelo; naranja = solo se sabe dónde estaba el dron;
            # azul = marca manual.
            color = '#2563eb' if p.manual else '#e74c3c' if p.located else '#e67e22'
            feats.append(_point(p.lon, p.lat, {
                'name': f'{"marca" if p.manual else "persona"} #{p.label}: {p.coords_text()}',
                'metadata': self._person_metadata(p),
                'ubicada': p.located,
                'manual': p.manual,
                'error_m': None if math.isnan(p.error_m) else round(p.error_m, 1),
                'score': round(p.score, 2),
                't': round(p.t, 2),
                'style': {'color': color, 'radius': 7, 'fillOpacity': 0.9},
            }))
        return {'type': 'FeatureCollection', 'features': feats}

    def _person_metadata(self, p: PersonSpot) -> dict:
        """Filas del tooltip del panel Map."""
        meta = {'hora': datetime.fromtimestamp(p.t).strftime('%H:%M:%S'),
                'score': f'{p.score:.2f}'}
        if not p.located:
            meta['lat dron'] = f'{p.drone_lat:.6f}'
            meta['lon dron'] = f'{p.drone_lon:.6f}'
            return meta
        meta['lat'] = f'{p.lat:.6f}'
        meta['lon'] = f'{p.lon:.6f}'
        if p.has_error():
            meta['error (1σ)'] = f'{p.error_m:.1f} m'
        if self.home is not None:
            north, east = _north_east_m(self.home, p.lat, p.lon)
            meta['desde home'] = (f'{abs(north):.0f} m {"N" if north >= 0 else "S"}, '
                                  f'{abs(east):.0f} m {"E" if east >= 0 else "O"}')
            meta['distancia a home'] = f'{math.hypot(north, east):.0f} m'
        return meta

    def rescue_waypoints(self) -> list:
        """Una entrada por persona. Sin ubicar: lat/lon en None, para que la
        posición del dron no se confunda con coordenadas de rescate."""
        out = []
        for p in self.persons:
            north = east = dist = None
            if p.located and self.home is not None:
                north, east = _north_east_m(self.home, p.lat, p.lon)
                dist = round(math.hypot(north, east), 1)
                north, east = round(north, 1), round(east, 1)
            out.append({
                'id': p.id,
                'etiqueta': p.label,
                'manual': p.manual,
                'ubicada': p.located,
                'lat': round(p.lat, 8) if p.located else None,
                'lon': round(p.lon, 8) if p.located else None,
                'error_m': round(p.error_m, 1) if p.has_error() else None,
                'norte_m': north,
                'este_m': east,
                'distancia_home_m': dist,
                'score': round(p.score, 2),
                't': round(p.t, 2),
                'drone_lat': round(p.drone_lat, 8),
                'drone_lon': round(p.drone_lon, 8),
            })
        return out

    def offset_from_home(self, lat, lon):
        """[norte, este] en metros desde home, redondeado; None sin home."""
        if self.home is None:
            return None
        north, east = _north_east_m(self.home, lat, lon)
        return [round(north, 2), round(east, 2)]

    def web_state(self) -> dict:
        """Lo que dibuja la página del mapa (web_map.py). Posiciones en metros
        desde home; `pos` de cada persona es la suya o, si no se ubicó, la del dron
        al verla (para dibujarla, nunca como coordenada de rescate)."""
        home = (None if self.home is None
                else {'lat': round(self.home.lat, 8), 'lon': round(self.home.lon, 8)})
        persons = []
        for w, p in zip(self.rescue_waypoints(), self.persons):
            persons.append({**w, 'pos': self.offset_from_home(p.lat, p.lon)})
        return {'home': home,
                'trail': [self.offset_from_home(p.lat, p.lon) for p in self.points],
                'length_m': round(self.total_m, 1),
                'persons': persons}

    def grid_features(self, step_m: float) -> list:
        """Líneas cada `step_m` centradas en home, cubriendo recorrido y personas."""
        if step_m <= 0 or self.home is None:
            return []
        offsets = [_north_east_m(self.home, s.lat, s.lon) for s in self.points + self.persons]
        norths = [n for n, _ in offsets] + [0.0]
        easts = [e for _, e in offsets] + [0.0]
        step = float(step_m)
        while True:
            n0, n1 = (math.floor(min(norths) / step) - 1, math.ceil(max(norths) / step) + 1)
            e0, e1 = (math.floor(min(easts) / step) - 1, math.ceil(max(easts) / step) + 1)
            if max(n1 - n0, e1 - e0) < MAX_GRID_LINES:
                break
            step *= 2.0
        feats = []
        for i in range(n0, n1 + 1):
            north = i * step
            feats.append(_grid_line(_axis_name(north, 'N', 'S'), i == 0, [
                _offset_lon_lat(self.home, north, e0 * step),
                _offset_lon_lat(self.home, north, e1 * step)]))
        for i in range(e0, e1 + 1):
            east = i * step
            feats.append(_grid_line(_axis_name(east, 'E', 'O'), i == 0, [
                _offset_lon_lat(self.home, n0 * step, east),
                _offset_lon_lat(self.home, n1 * step, east)]))
        return feats

    def to_kml(self, name='Vuelo YOLO') -> str:
        def coords(p):
            alt = 0.0 if math.isnan(p.alt_msl) else p.alt_msl
            return f'{p.lon:.8f},{p.lat:.8f},{alt:.2f}'

        out = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
            f'<name>{escape(name)}</name>',
            '<Style id="ruta"><LineStyle><color>ffff901e</color><width>4</width>'
            '</LineStyle></Style>',
            '<Style id="persona"><IconStyle><color>ff3c4ce7</color><Icon><href>'
            'http://maps.google.com/mapfiles/kml/shapes/man.png</href></Icon>'
            '</IconStyle></Style>',
            '<Style id="home"><IconStyle><color>ff71cc2e</color><Icon><href>'
            'http://maps.google.com/mapfiles/kml/shapes/homegardenbusiness.png</href>'
            '</Icon></IconStyle></Style>',
        ]
        if len(self.points) >= 2:
            out.append(
                '<Placemark><name>Recorrido</name><styleUrl>#ruta</styleUrl>'
                '<LineString><tessellate>1</tessellate><altitudeMode>absolute</altitudeMode>'
                '<coordinates>' + ' '.join(coords(p) for p in self.points)
                + '</coordinates></LineString></Placemark>')
        if self.home is not None:
            out.append('<Placemark><name>Home</name><styleUrl>#home</styleUrl>'
                       f'<Point><coordinates>{self.home.lon:.8f},{self.home.lat:.8f},0'
                       '</coordinates></Point></Placemark>')
        for p in self.persons:
            out.append(
                f'<Placemark><name>{"Marca" if p.manual else "Persona"} #{p.label}</name>'
                f'<description>{p.coords_text()}, score {p.score:.2f}, t={p.t:.1f}s</description>'
                '<styleUrl>#persona</styleUrl>'
                f'<Point><coordinates>{p.lon:.8f},{p.lat:.8f},0</coordinates></Point>'
                '</Placemark>')
        out.append('</Document></kml>')
        return '\n'.join(out)

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(['t', 'lat', 'lon', 'alt_msl', 'alt_rel', 'heading_deg'])
        for p in self.points:
            w.writerow([f'{p.t:.3f}', f'{p.lat:.8f}', f'{p.lon:.8f}', f'{p.alt_msl:.2f}',
                        _fmt(p.alt_rel), _fmt(p.heading_deg)])
        return buf.getvalue()

    def persons_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(['id', 't', 'lat', 'lon', 'ubicada', 'error_m', 'score',
                    'drone_lat', 'drone_lon', 'manual', 'etiqueta'])
        for p in self.persons:
            # id: siempre el número (negativo en las marcas manuales); la etiqueta
            # que se ve en el mapa (1, 2, M1...) va al final, sin correr columnas.
            w.writerow([p.id, f'{p.t:.3f}', f'{p.lat:.8f}', f'{p.lon:.8f}',
                        int(p.located), _fmt(p.error_m), f'{p.score:.3f}',
                        f'{p.drone_lat:.8f}', f'{p.drone_lon:.8f}', int(p.manual), p.label])
        return buf.getvalue()

    def geojson_str(self, grid_step_m=0.0) -> str:
        """Mensaje en vivo para el mapa. La grilla va solo acá, no en trail.geojson,
        y primero para quedar dibujada debajo de todo."""
        gj = self.to_geojson()
        gj['features'] = self.grid_features(grid_step_m) + gj['features']
        return json.dumps(gj, separators=(',', ':'))


def _point(lon, lat, props):
    return {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
            'properties': props}


def _line(coords, props):
    return {'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': coords},
            'properties': props}


def _search_ring(p: PersonSpot):
    """Anillo de 2 sigma, como línea cerrada: el panel Map documenta el estilo
    de líneas, no el de polígonos."""
    radius = 2.0 * p.error_m
    coords = []
    for i in range(RING_SEGMENTS + 1):
        a = 2.0 * math.pi * (i % RING_SEGMENTS) / RING_SEGMENTS
        coords.append(_offset_lon_lat(p, radius * math.cos(a), radius * math.sin(a)))
    return _line(coords, {
        'name': f'zona de búsqueda (2σ) persona #{p.id}: {radius:.1f} m de radio',
        'style': {'color': '#e74c3c', 'weight': 2, 'opacity': 0.8},
    })


def _grid_line(name, is_axis, coords):
    return _line(coords, {
        'name': name,
        'style': {'color': '#7f8c8d', 'weight': 2 if is_axis else 1,
                  'opacity': 0.8 if is_axis else 0.4},
    })


def _axis_name(metres, positive, negative):
    if metres == 0:
        return f'0 m {positive}{negative} (home)'
    return f'{abs(metres):.0f} m {positive if metres > 0 else negative}'


def _fmt(v):
    return '' if v is None or math.isnan(v) else f'{v:.2f}'

