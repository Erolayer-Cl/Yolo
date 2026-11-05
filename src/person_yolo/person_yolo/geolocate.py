"""Dónde está la persona en el suelo (lat/lon), no solo dónde estaba el dron.

Se traza el rayo que sale de la cámara por el centro de la caja de la persona
y se lo corta con un plano horizontal a la altura del centro de una persona
(~0.9 m sobre el suelo). La distancia al suelo sale del lidar hacia abajo (o,
sin lidar, de la altura sobre el despegue: supone suelo plano).

Adaptado de Mision_1/src/patrol_core/patrol_core/ground_projection.py (mismo
método: centro de la caja, corte por incidencia, fórmula de incertidumbre),
pero trabajando directo en la convención de PX4 para no convertir marcos:
  - mundo NED (x = Norte, y = Este, z = abajo),
  - cuerpo FRD (x = adelante, y = derecha, z = abajo),
  - actitud = cuaternión (w, x, y, z) que lleva FRD a NED, tal cual lo publica
    /fmu/out/vehicle_attitude (gps_node lo reenvía en /drone/attitude).
Cámara: modelo pinhole con FOV horizontal y montaje inclinado `pitch_deg`
(negativo = hacia abajo, misma convención que el `Pitch` de AirSim).

Python puro, testeable con pytest.
"""
import math
from dataclasses import dataclass

from person_yolo.trail import offset_to_latlon


@dataclass
class Camera:
    hfov_deg: float                 # FOV horizontal (FOV_Degrees en AirSim)
    pitch_deg: float = 0.0          # montaje: -30 = 30° hacia abajo; -90 = nadir


def quat_rotate(q, v):
    """Rota el vector v con el cuaternión unitario q = (w, x, y, z)."""
    w, ux, uy, uz = q
    vx, vy, vz = v
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    ccx, ccy, ccz = uy * cz - uz * cy, uz * cx - ux * cz, ux * cy - uy * cx
    return (vx + 2.0 * (w * cx + ccx), vy + 2.0 * (w * cy + ccy), vz + 2.0 * (w * cz + ccz))


def cos_tilt(q):
    """Coseno de la inclinación del cuerpo respecto de la vertical: la
    componente 'abajo' del eje z del cuerpo. Sirve para pasar la lectura de
    un lidar fijo al cuerpo (inclinado) a altura vertical."""
    _, x, y, _ = q
    return 1.0 - 2.0 * (x * x + y * y)


def pixel_ray_body(u, v, width, height, cam: Camera):
    """Pixel -> rayo unitario en el cuerpo FRD."""
    fx = (width / 2.0) / math.tan(math.radians(cam.hfov_deg) / 2.0)
    # Marco óptico (x derecha, y abajo, z adelante) -> FRD de la cámara sin inclinar.
    ox, oy = (u - width / 2.0) / fx, (v - height / 2.0) / fx
    x, y, z = 1.0, ox, oy
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    # Montaje inclinado: rotación sobre el eje y del cuerpo (pitch + = nariz arriba).
    p = math.radians(cam.pitch_deg)
    return (x * math.cos(p) + z * math.sin(p), y, -x * math.sin(p) + z * math.cos(p))


def touches_edge(box, width, height, margin_px=3.0):
    """La caja toca el borde de la imagen: la persona está cortada y el centro
    de la caja no es el suyo, así que no sirve para mejorar su posición."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return (x1 <= margin_px or y1 <= margin_px
            or x2 >= width - margin_px or y2 >= height - margin_px)


def locate(u, v, width, height, cam: Camera, q_ned_frd, agl_m, lat, lon,
           person_height_m=0.9, max_incidence_deg=65.0, sigma_pix=2.0,
           sigma_att_deg=1.0, sigma_alt=0.3, sigma_gps=2.5):
    """Posición en el suelo de la persona cuyo centro de caja está en (u, v).

    `agl_m`: altura de la cámara sobre el suelo. Devuelve None si el rayo no
    cruza el plano por debajo (mira al cielo) o si llega más rasante que
    `max_incidence_deg` desde la vertical: ahí el error crece sin control.
    Si no, un dict con lat, lon, north_m, east_m, dist_m, incidence_deg y
    error_m (1 sigma, combina píxel, actitud, altura y GPS del dron). También
    None con datos inválidos (NaN, actitud sin normalizar): con ellos saldría
    una posición falsa o un NaN que rompe el GeoJSON del mapa.
    """
    if not all(math.isfinite(v) for v in (agl_m, lat, lon, *q_ned_frd)):
        return None
    if abs(math.sqrt(sum(c * c for c in q_ned_frd)) - 1.0) > 0.1:
        return None
    h = agl_m - person_height_m
    if h <= 0.1:
        return None
    dn, de, dd = quat_rotate(q_ned_frd, pixel_ray_body(u, v, width, height, cam))
    if dd <= 1e-6:
        return None
    incidence = math.acos(max(-1.0, min(1.0, dd)))
    if incidence > math.radians(max_incidence_deg):
        return None
    t = h / dd
    north, east = t * dn, t * de
    plat, plon = offset_to_latlon(lat, lon, north, east)

    # Incertidumbre (ver Mision_1 sigma_ground): el ruido angular se proyecta
    # con 1/cos² de la incidencia y el error de altura con su tangente.
    fx = (width / 2.0) / math.tan(math.radians(cam.hfov_deg) / 2.0)
    c2 = math.cos(incidence) ** 2
    e_pix = h * sigma_pix / (fx * c2)
    e_att = h * math.radians(sigma_att_deg) / c2
    e_alt = math.tan(incidence) * sigma_alt
    error = math.sqrt(e_pix ** 2 + e_att ** 2 + e_alt ** 2 + sigma_gps ** 2)
    return {'lat': plat, 'lon': plon, 'north_m': north, 'east_m': east,
            'dist_m': math.hypot(north, east), 'incidence_deg': math.degrees(incidence),
            'error_m': error}
