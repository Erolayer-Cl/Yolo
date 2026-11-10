#!/usr/bin/env python3
"""Deja claro por dónde pasó el dron y dónde vio personas.

Entradas:  /drone/gps (NavSatFix), /drone/pose (PoseStamped ENU),
           /yolo/person_alert (String JSON de person_detector, una por persona),
           /yolo/person_fix (mismo JSON, cada vez que mejora su posición)
           /yolo/person_merge (dos marcas eran la misma persona: se borra una)
Salidas (1 Hz):
  /drone/trail_geojson       foxglove_msgs/GeoJSON  <- panel Map: línea azul del
                             recorrido, punto verde = home, rojos = personas. El dron
                             en vivo es /drone/gps. (Requiere ros-<distro>-foxglove-msgs;
                             si no está, el resto funciona igual y se avisa.)
  /rescue/waypoints          String JSON <- panel Raw Messages: coordenadas de cada persona
Web (web_map.py): http://<host>:<web_port>/ mapa de rescate en el navegador, con la
  lista de personas y sus coordenadas GPS. En SITL las coordenadas las simula
  Unreal (OriginGeopoint de AirSim); en el dron real, el GPS. web_port 0 = apagada.
  Botón "Marcar aquí": agrega una marca manual (M1, M2...) en la posición del dron
  en ese momento, p. ej. al ver a alguien en el FPV que el detector no marcó bien.
  Es solo un punto en el mapa: no comanda al dron. web_marks: false lo desactiva.
Archivos (cada save_period_sec y al cerrar), en output_dir/<fecha_hora>/:
  trail.geojson, trail.kml (Google Earth), trail.csv, personas.csv (una fila por
  persona: su lat/lon en el suelo y error, o la del dron si no se pudo ubicar)
Servicio:
  /drone/trail/reset (std_srvs/Trigger): guarda lo actual y empieza un recorrido nuevo.

La lógica (decimación, export) está en trail.py, testeada sin ROS.
"""
import json
import math
import os
import queue
import time
from datetime import datetime

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger

from person_yolo.node_main import run
from person_yolo.overlay import enu_quat_to_heading_deg
from person_yolo.trail import Trail, TrailPoint, spot_from_alert
from person_yolo.web_map import MapServer

try:
    from foxglove_msgs.msg import GeoJSON
except ImportError:          # paquete opcional
    GeoJSON = None


def _stamp_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def _write_atomic(path, text):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
        # Sin fsync, un corte de batería en la Jetson puede dejar el archivo vacío.
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _num(v, nd):
    """Redondea, o None si no es un número finito (JSON no admite NaN)."""
    return round(v, nd) if v is not None and math.isfinite(v) else None


class TrailNode(Node):

    def __init__(self):
        super().__init__('trail_node')

        self.declare_parameter('min_step_m', 1.0)
        self.declare_parameter('max_points', 5000)
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('output_dir', '~/yolo_flights')
        self.declare_parameter('save_period_sec', 10.0)
        self.declare_parameter('grid_step_m', 50.0)    # grilla del mapa; 0 = sin grilla
        self.declare_parameter('web_port', 8081)        # mapa en el navegador; 0 = apagado
        self.declare_parameter('web_bind_address', '0.0.0.0')
        self.declare_parameter('web_live_rate_hz', 5.0)  # posición del dron en la web
        # Botón "Marcar aquí" de la web (marca en la posición del dron). La página no
        # tiene contraseña: cualquiera en la red puede agregar o borrar marcas manuales
        # (nunca las del detector, y nunca comanda al dron).
        self.declare_parameter('web_marks', True)
        self._web_queue = queue.SimpleQueue()   # pedidos de la web -> hilo de ROS

        self.grid_step_m = float(self.get_parameter('grid_step_m').value)
        self.output_root = os.path.expanduser(self.get_parameter('output_dir').value)
        self._fix = None                     # último GPS válido, para la web
        self._in_view = None                 # (personas en cámara, instante), para la web
        self._new_trail()

        # Latched (TRANSIENT_LOCAL): si abrís Foxglove a mitad del vuelo, el
        # mapa recibe el recorrido completo enseguida, sin esperar el próximo tick.
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_geojson = (self.create_publisher(GeoJSON, '/drone/trail_geojson', latched)
                            if GeoJSON is not None else None)
        self.pub_waypoints = self.create_publisher(String, '/rescue/waypoints', latched)
        if GeoJSON is None:
            self.get_logger().warn(
                'foxglove_msgs no está instalado: no se publica /drone/trail_geojson '
                '(la línea del recorrido en el panel Map). Instalalo con\n'
                '  sudo apt install ros-$ROS_DISTRO-foxglove-msgs\n'
                'y recompilá. Los archivos KML/GeoJSON/CSV se siguen guardando.')

        self.create_subscription(NavSatFix, '/drone/gps', self._on_gps, 10)
        self.create_subscription(PoseStamped, '/drone/pose', self._on_pose, 10)
        self.create_subscription(String, '/yolo/person_alert', self._on_alert, 50)
        self.create_subscription(String, '/yolo/person_fix', self._on_alert, 50)
        self.create_subscription(String, '/yolo/person_merge', self._on_merge, 50)
        self.create_subscription(Int32, '/yolo/person_count', self._on_count, 10)
        self.create_service(Trigger, '/drone/trail/reset', self._on_reset)

        rate = float(self.get_parameter('publish_rate_hz').value)
        self.create_timer(1.0 / max(rate, 0.1), self._publish)
        save = float(self.get_parameter('save_period_sec').value)
        if save > 0:
            self.create_timer(save, self.save)

        self.web = None
        port = int(self.get_parameter('web_port').value)
        if port > 0:
            marks = bool(self.get_parameter('web_marks').value)
            try:
                self.web = MapServer(port, self.get_parameter('web_bind_address').value,
                                     on_post=self._on_web_post if marks else None)
            except OSError as exc:            # p.ej. puerto ocupado: el resto sigue igual
                self.get_logger().error(f'No se pudo abrir el mapa web en :{port}: {exc}')
        if self.web is not None:
            live = float(self.get_parameter('web_live_rate_hz').value)
            self.create_timer(1.0 / max(live, 0.5), self._web_live)
            self.create_timer(0.1, self._apply_web)
            self._web_state()
            self.get_logger().info(f'Mapa de rescate en http://localhost:{port}/')

        self.get_logger().info(f'trail_node listo. Recorridos en {self.output_root}/')

    def _new_trail(self):
        self.trail = Trail(
            min_step_m=float(self.get_parameter('min_step_m').value),
            max_points=int(self.get_parameter('max_points').value))
        self.pose = None
        self.flight_dir = None               # se crea al primer guardado con datos

    # ------------------------------------------------------------- entradas
    def _on_pose(self, msg: PoseStamped):
        self.pose = msg

    def _on_gps(self, msg: NavSatFix):
        if msg.status.status < NavSatStatus.STATUS_FIX:
            return
        if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
            return
        alt_rel = heading = float('nan')
        if self.pose is not None:
            alt_rel = self.pose.pose.position.z
            q = self.pose.pose.orientation
            heading = enu_quat_to_heading_deg(q.z, q.w)
        self.trail.add(TrailPoint(_stamp_sec(msg.header.stamp), msg.latitude, msg.longitude,
                                  msg.altitude, alt_rel, heading))
        self._fix = (msg.latitude, msg.longitude, alt_rel, heading, time.monotonic())

    def _on_count(self, msg: Int32):
        self._in_view = (int(msg.data), time.monotonic())

    def _on_alert(self, msg: String):
        try:
            al = json.loads(msg.data)
        except ValueError:
            return
        spot = spot_from_alert(al)
        if spot is None:
            self.get_logger().warn('Persona sin GPS válido: no se puede marcar en el '
                                   f'mapa ({msg.data[:120]})')
            return
        self.trail.add_person(spot)
        self.get_logger().info(f'Persona #{spot.id}: {spot.coords_text()}')
        self._web_state()                     # en la web, sin esperar al próximo tick

    def _on_merge(self, msg: String):
        try:
            m = json.loads(msg.data)
            rid, rt, keep = int(m['removed_id']), float(m['removed_t']), int(m['id'])
        except (KeyError, TypeError, ValueError):
            return
        if self.trail.remove_person(rid, rt):
            self.get_logger().info(f'Persona #{rid} era la #{keep}: se borra del mapa')
            self._web_state()

    # ------------------------------------------------------- marcas manuales
    def _on_web_post(self, path, data):
        """Hilo del servidor web: valida, encola y responde. No toca self.trail."""
        if path == '/api/mark':
            fix = self._fix                      # tupla: se lee entera (atómico)
            if fix is None or time.monotonic() - fix[4] > 3.0:
                return 409, {'error': 'sin GPS del dron: no se puede marcar'}
            lat, lon = fix[0], fix[1]
            self._web_queue.put(('add', time.time(), lat, lon))
            return 200, {'ok': True, 'lat': round(lat, 8), 'lon': round(lon, 8)}
        if path == '/api/mark/delete':
            try:
                pid = int(data['id'])
            except (KeyError, TypeError, ValueError):
                return 400, {'error': 'falta "id"'}
            if pid >= 0:
                return 400, {'error': 'solo se borran marcas manuales'}
            self._web_queue.put(('del', pid))
            return 200, {'ok': True}
        return 404, {'error': 'no existe'}

    def _apply_web(self):
        """Hilo de ROS: aplica los pedidos de la web."""
        changed = False
        while True:
            try:
                req = self._web_queue.get_nowait()
            except queue.Empty:
                break
            if req[0] == 'add':
                spot = self.trail.add_mark(*req[1:])
                self.get_logger().warn(f'MARCA MANUAL {spot.label} en {spot.lat:.6f}, '
                                       f'{spot.lon:.6f} (posición del dron)')
                changed = True
            elif req[0] == 'del' and self.trail.remove_mark(req[1]):
                self.get_logger().info(f'Marca M{-req[1]} borrada desde la web')
                changed = True
        if changed:
            self._publish()

    def _on_reset(self, request, response):
        self.save()
        old = self.flight_dir
        self._new_trail()
        self._publish()
        response.success = True
        response.message = f'recorrido nuevo (el anterior quedó en {old})'
        self.get_logger().info('/drone/trail/reset: ' + response.message)
        return response

    # ------------------------------------------------------------- salidas
    def _publish(self):
        if self.pub_geojson is not None:
            self.pub_geojson.publish(GeoJSON(geojson=self.trail.geojson_str(self.grid_step_m)))
        self.pub_waypoints.publish(String(data=json.dumps(
            {'personas': self.trail.rescue_waypoints()}, indent=1, ensure_ascii=False)))
        self._web_state()

    def _web_state(self):
        if self.web is None:
            return
        self.web.set('/api/state',
                     json.dumps(self.trail.web_state(), separators=(',', ':')).encode())
        self.web.set('/personas.csv', self.trail.persons_csv().encode())

    def _web_live(self):
        live = {'lat': None, 'lon': None, 'alt_rel': None, 'heading_deg': None,
                'age_s': None, 'ne': None, 'in_view': None}
        # El detector publica el conteo en cada inferencia: si hace 2 s que no
        # llega, el número ya no dice nada (cámara o detector caídos).
        if self._in_view is not None and time.monotonic() - self._in_view[1] < 2.0:
            live['in_view'] = self._in_view[0]
        if self._fix is not None:
            lat, lon, alt_rel, heading, t = self._fix
            live.update(lat=round(lat, 8), lon=round(lon, 8), alt_rel=_num(alt_rel, 2),
                        heading_deg=_num(heading, 1), age_s=round(time.monotonic() - t, 1),
                        ne=self.trail.offset_from_home(lat, lon))
        self.web.set('/api/live', json.dumps(live).encode())

    def destroy_node(self):
        if self.web is not None:
            self.web.stop()
        super().destroy_node()

    def save(self):
        if not self.trail.dirty or not self.trail.points:
            return
        if self.flight_dir is None:
            name = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.flight_dir = os.path.join(self.output_root, name)
            os.makedirs(self.flight_dir, exist_ok=True)
            self.get_logger().info(f'Guardando recorrido en {self.flight_dir}/')
        try:
            d = self.flight_dir
            _write_atomic(os.path.join(d, 'trail.geojson'),
                          json.dumps(self.trail.to_geojson(), indent=1))
            _write_atomic(os.path.join(d, 'trail.kml'),
                          self.trail.to_kml(name=f'Vuelo {os.path.basename(d)}'))
            _write_atomic(os.path.join(d, 'trail.csv'), self.trail.to_csv())
            _write_atomic(os.path.join(d, 'personas.csv'), self.trail.persons_csv())
            self.trail.dirty = False
        except OSError as exc:
            self.get_logger().error(f'No se pudo guardar el recorrido: {exc}')


def main(args=None):
    # on_exit: último guardado del recorrido al cerrar (Ctrl-C).
    run(TrailNode, args, on_exit=lambda n: n.save())


if __name__ == '__main__':
    main()
