#!/usr/bin/env python3
"""Deja claro por dónde pasó el dron y dónde vio personas.

Entradas:  /drone/gps (NavSatFix), /drone/pose (PoseStamped ENU),
           /yolo/person_events (String JSON de person_detector)
Salidas (1 Hz):
  /drone/trail_geojson       foxglove_msgs/GeoJSON  <- panel Map: línea azul del
                             recorrido, punto verde = home, rojos = personas,
                             amarillo = dron. (Requiere ros-<distro>-foxglove-msgs;
                             si no está, el resto funciona igual y se avisa.)
Archivos (cada save_period_sec y al cerrar), en output_dir/<fecha_hora>/:
  trail.geojson, trail.kml (Google Earth), trail.csv, detections.csv
Servicio:
  /drone/trail/reset (std_srvs/Trigger): guarda lo actual y empieza un recorrido nuevo.

La lógica (decimación, export) está en trail.py, testeada sin ROS.
"""
import json
import math
import os
from datetime import datetime

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import String
from std_srvs.srv import Trigger

from person_yolo.node_main import run
from person_yolo.overlay import enu_quat_to_heading_deg
from person_yolo.trail import Trail, TrailPoint

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
    os.replace(tmp, path)


class TrailNode(Node):

    def __init__(self):
        super().__init__('trail_node')

        self.declare_parameter('min_step_m', 1.0)
        self.declare_parameter('max_points', 5000)
        self.declare_parameter('det_merge_m', 5.0)
        self.declare_parameter('det_merge_sec', 10.0)
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('output_dir', '~/yolo_flights')
        self.declare_parameter('save_period_sec', 10.0)

        self.output_root = os.path.expanduser(self.get_parameter('output_dir').value)
        self._new_trail()

        # Latched (TRANSIENT_LOCAL): si abrís Foxglove a mitad del vuelo, el
        # mapa recibe el recorrido completo enseguida, sin esperar el próximo tick.
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_geojson = (self.create_publisher(GeoJSON, '/drone/trail_geojson', latched)
                            if GeoJSON is not None else None)
        if GeoJSON is None:
            self.get_logger().warn(
                'foxglove_msgs no está instalado: no se publica /drone/trail_geojson '
                '(la línea del recorrido en el panel Map). Instalalo con\n'
                '  sudo apt install ros-$ROS_DISTRO-foxglove-msgs\n'
                'y recompilá. Los archivos KML/GeoJSON/CSV se siguen guardando.')

        self.create_subscription(NavSatFix, '/drone/gps', self._on_gps, 10)
        self.create_subscription(PoseStamped, '/drone/pose', self._on_pose, 10)
        self.create_subscription(String, '/yolo/person_events', self._on_event, 50)
        self.create_service(Trigger, '/drone/trail/reset', self._on_reset)

        rate = float(self.get_parameter('publish_rate_hz').value)
        self.create_timer(1.0 / max(rate, 0.1), self._publish)
        save = float(self.get_parameter('save_period_sec').value)
        if save > 0:
            self.create_timer(save, self.save)

        self.get_logger().info(f'trail_node listo. Recorridos en {self.output_root}/')

    def _new_trail(self):
        self.trail = Trail(
            min_step_m=float(self.get_parameter('min_step_m').value),
            max_points=int(self.get_parameter('max_points').value),
            det_merge_m=float(self.get_parameter('det_merge_m').value),
            det_merge_sec=float(self.get_parameter('det_merge_sec').value))
        self.current = None
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
        self.current = TrailPoint(_stamp_sec(msg.header.stamp), msg.latitude, msg.longitude,
                                  msg.altitude, alt_rel, heading)
        self.trail.add(self.current)

    def _on_event(self, msg: String):
        try:
            ev = json.loads(msg.data)
        except ValueError:
            return
        count = int(ev.get('count', 0))
        if count <= 0:
            return
        lat, lon = ev.get('lat'), ev.get('lon')
        if lat is not None and lon is not None:
            alt_rel = ev.get('alt_rel')
            n_before = len(self.trail.detections)
            self.trail.add_detection(float(ev.get('t', 0.0)), lat, lon,
                                     float('nan') if alt_rel is None else alt_rel,
                                     count, float(ev.get('max_score', 0.0)))
            if len(self.trail.detections) > n_before:
                self.get_logger().info(
                    f'Personas ({count}) vistas en lat={lat:.6f} lon={lon:.6f} '
                    f'-> punto #{len(self.trail.detections)} en el mapa')

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
        if self.pub_geojson is not None:          # siempre: el punto del dron se mueve
            self.pub_geojson.publish(GeoJSON(geojson=self.trail.geojson_str(self.current)))

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
            _write_atomic(os.path.join(d, 'detections.csv'), self.trail.detections_csv())
            self.trail.dirty = False
        except OSError as exc:
            self.get_logger().error(f'No se pudo guardar el recorrido: {exc}')


def main(args=None):
    # on_exit: último guardado del recorrido al cerrar (Ctrl-C).
    run(TrailNode, args, on_exit=lambda n: n.save())


if __name__ == '__main__':
    main()
