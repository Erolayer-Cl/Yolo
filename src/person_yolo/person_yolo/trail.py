"""Recorrido del dron y lugares donde vio personas.

Python puro (sin rclpy): decimación por distancia y export a GeoJSON (panel
Map de Foxglove), KML (Google Earth) y CSV. Lo usa `trail_node`.
"""
import csv
import io
import json
import math
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

EARTH_R = 6371008.8


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(min(1.0, a)))


@dataclass
class TrailPoint:
    t: float                # segundos (stamp ROS)
    lat: float
    lon: float
    alt_msl: float
    alt_rel: float          # altura sobre home; NaN si no hay pose local
    heading_deg: float      # 0 = Norte, horario; NaN si no hay


@dataclass
class DetectionSpot:
    t_first: float
    t_last: float
    lat: float
    lon: float
    alt_rel: float
    max_count: int
    max_score: float
    hits: int = 1           # cuántos frames con personas se fusionaron acá


@dataclass
class Trail:
    """Recorrido decimado: un punto nuevo solo si el dron se movió.

    Con el dron quieto en hover el GPS sigue llegando a 5-10 Hz; sin decimar
    la línea crecería sin parar y el mensaje al mapa se volvería enorme.
    """
    min_step_m: float = 1.0
    max_points: int = 5000
    det_merge_m: float = 5.0        # eventos de detección a menos de esto...
    det_merge_sec: float = 10.0     # ...y dentro de este tiempo se fusionan
    points: list = field(default_factory=list)
    detections: list = field(default_factory=list)
    home: TrailPoint = None
    total_m: float = 0.0
    dirty: bool = False

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

    def add_detection(self, t, lat, lon, alt_rel, count, max_score) -> DetectionSpot:
        """Registra que el dron vio `count` personas en (lat, lon).

        A 10 Hz, sobrevolar a una persona genera decenas de eventos seguidos;
        se fusionan en un solo punto si están cerca en espacio y tiempo.
        """
        if self.detections:
            last = self.detections[-1]
            if (t - last.t_last <= self.det_merge_sec
                    and haversine_m(last.lat, last.lon, lat, lon) <= self.det_merge_m):
                last.t_last = t
                last.hits += 1
                last.max_count = max(last.max_count, int(count))
                last.max_score = max(last.max_score, float(max_score))
                self.dirty = True
                return last
        spot = DetectionSpot(t, t, lat, lon, alt_rel, int(count), float(max_score))
        self.detections.append(spot)
        self.dirty = True
        return spot

    # ------------------------------------------------------------- exports
    def to_geojson(self, current: TrailPoint = None) -> dict:
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
        for i, d in enumerate(self.detections, 1):
            feats.append(_point(d.lon, d.lat, {
                'name': f'personas #{i}',
                'personas_max': d.max_count,
                'score_max': round(d.max_score, 2),
                'alt_rel_m': None if math.isnan(d.alt_rel) else round(d.alt_rel, 1),
                't_first': round(d.t_first, 2),
                't_last': round(d.t_last, 2),
                'frames': d.hits,
                'style': {'color': '#e74c3c', 'radius': 7, 'fillOpacity': 0.9},
            }))
        if current is not None:
            feats.append(_point(current.lon, current.lat, {
                'name': 'dron',
                'style': {'color': '#f1c40f', 'radius': 6, 'fillOpacity': 1.0},
            }))
        return {'type': 'FeatureCollection', 'features': feats}

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
        for i, d in enumerate(self.detections, 1):
            out.append(
                f'<Placemark><name>Personas #{i} ({d.max_count})</name>'
                f'<description>max {d.max_count} personas, score {d.max_score:.2f}, '
                f'{d.hits} frames, t={d.t_first:.1f}-{d.t_last:.1f}s</description>'
                '<styleUrl>#persona</styleUrl>'
                f'<Point><coordinates>{d.lon:.8f},{d.lat:.8f},0</coordinates></Point>'
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

    def detections_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(['t_first', 't_last', 'lat', 'lon', 'alt_rel', 'max_count',
                    'max_score', 'frames'])
        for d in self.detections:
            w.writerow([f'{d.t_first:.3f}', f'{d.t_last:.3f}', f'{d.lat:.8f}',
                        f'{d.lon:.8f}', _fmt(d.alt_rel), d.max_count,
                        f'{d.max_score:.3f}', d.hits])
        return buf.getvalue()

    def geojson_str(self, current=None) -> str:
        return json.dumps(self.to_geojson(current), separators=(',', ':'))


def _point(lon, lat, props):
    return {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
            'properties': props}


def _fmt(v):
    return '' if v is None or math.isnan(v) else f'{v:.2f}'

