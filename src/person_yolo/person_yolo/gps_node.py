#!/usr/bin/env python3
"""Dónde está el dron: traduce PX4 (uXRCE-DDS) a tópicos ROS estándar.

  /drone/gps     sensor_msgs/NavSatFix   <- panel Map de Foxglove
  /drone/pose    geometry_msgs/PoseStamped (frame `odom`, ENU, z = altura sobre home)
  /drone/status  std_msgs/String (JSON: armado, modo, satélites, fix, altura)
  /drone/attitude geometry_msgs/QuaternionStamped: actitud completa tal cual
                 PX4 (frame `ned`: cuerpo FRD -> NED, w,x,y,z). La usa
                 person_detector para ubicar personas en el suelo (geolocate.py).
  /drone/agl     sensor_msgs/Range: altura sobre el terreno del EKF de PX4, solo
                 si la mide un lidar (dist_bottom con sensor de rango). Vertical.

Funciona igual en SITL y en el dron real: en ambos casos PX4 publica
/fmu/out/* a través del MicroXRCEAgent. Desde PX4 1.16 los mensajes
versionados llevan sufijo _vN en el nombre del tópico (defaults = PX4 v1.17.x);
si no coincide, _check_alive avisa con el nombre correcto. Este nodo solo LEE de PX4; nunca
publica en /fmu/in/*, así que no interfiere con el vuelo manual.

Base: gps_bridge.py de Mision_1/patrol_bridge (corregido respecto de la beta).
"""
import json
import math
import re

from geometry_msgs.msg import PoseStamped, QuaternionStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix, NavSatStatus, Range
from std_msgs.msg import String

from px4_msgs.msg import SensorGps, VehicleAttitude, VehicleLocalPosition, VehicleStatus

from person_yolo.node_main import run

NAV_STATES = {
    0: 'MANUAL', 1: 'ALTCTL', 2: 'POSCTL', 3: 'AUTO_MISSION', 4: 'AUTO_LOITER',
    5: 'AUTO_RTL', 6: 'POSITION_SLOW', 10: 'ACRO', 12: 'DESCEND', 13: 'TERMINATION',
    14: 'OFFBOARD', 15: 'STAB', 17: 'AUTO_TAKEOFF', 18: 'AUTO_LAND',
    19: 'AUTO_FOLLOW_TARGET', 20: 'AUTO_PRECLAND', 21: 'ORBIT',
}
FIX_NAMES = {0: 'none', 1: 'none', 2: '2D', 3: '3D', 4: 'DGPS', 5: 'RTK_float',
             6: 'RTK_fixed', 8: 'extrapolated'}


def ned_heading_to_enu_quat(heading):
    """Yaw PX4 (0 = Norte, horario) -> cuaternión ENU (0 = Este, antihorario)."""
    yaw = math.pi / 2.0 - heading
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class GpsNode(Node):

    def __init__(self):
        super().__init__('gps_node')

        self.declare_parameter('gps_topic', '/fmu/out/vehicle_gps_position')
        self.declare_parameter('local_position_topic',
                               '/fmu/out/vehicle_local_position_v1')
        self.declare_parameter('status_topic', '/fmu/out/vehicle_status_v1')
        self.declare_parameter('attitude_topic', '/fmu/out/vehicle_attitude')
        self.declare_parameter('pose_rate_hz', 20.0)    # local_position llega a ~100 Hz
        self.declare_parameter('status_rate_hz', 1.0)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        rate = float(self.get_parameter('pose_rate_hz').value)
        self.pose_min_period_ns = int(1e9 / rate) if rate > 0 else 0

        # PX4 publica en BEST_EFFORT: un suscriptor RELIABLE no matchea y no
        # llega nada, sin ningún error visible.
        px4_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE,
                             history=HistoryPolicy.KEEP_LAST, depth=5)

        self.pub_gps = self.create_publisher(NavSatFix, '/drone/gps', 10)
        self.pub_pose = self.create_publisher(PoseStamped, '/drone/pose', 10)
        self.pub_status = self.create_publisher(String, '/drone/status', 10)
        self.pub_att = self.create_publisher(QuaternionStamped, '/drone/attitude', 10)
        self.pub_agl = self.create_publisher(Range, '/drone/agl', 10)

        self.create_subscription(SensorGps, self.get_parameter('gps_topic').value,
                                 self._on_gps, px4_qos)
        self.create_subscription(VehicleLocalPosition,
                                 self.get_parameter('local_position_topic').value,
                                 self._on_local, px4_qos)
        self.create_subscription(VehicleStatus, self.get_parameter('status_topic').value,
                                 self._on_status, px4_qos)
        self.create_subscription(VehicleAttitude, self.get_parameter('attitude_topic').value,
                                 self._on_attitude, px4_qos)

        srate = float(self.get_parameter('status_rate_hz').value)
        if srate > 0:
            self.create_timer(1.0 / srate, self._publish_status)

        self._gps = None
        self._local = None
        self._vstatus = None
        self._att = None
        self._last_pose_ns = 0
        self._last_att_ns = 0
        self._warned = set()
        self.create_timer(5.0, self._check_alive)

        self.get_logger().info('gps_node listo, esperando /fmu/out/* de PX4...')

    # ------------------------------------------------------------- callbacks
    def _on_gps(self, msg: SensorGps):
        self._gps = msg
        out = NavSatFix()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.base_frame
        out.status.service = NavSatStatus.SERVICE_GPS
        # 8 = extrapolado (ver FIX_NAMES): no es una medición, cuenta como sin fix.
        if msg.fix_type in (5, 6):
            out.status.status = NavSatStatus.STATUS_GBAS_FIX
        elif msg.fix_type in (2, 3, 4):
            out.status.status = NavSatStatus.STATUS_FIX
        else:
            out.status.status = NavSatStatus.STATUS_NO_FIX
        out.latitude = msg.latitude_deg
        out.longitude = msg.longitude_deg
        out.altitude = msg.altitude_msl_m
        if math.isfinite(msg.eph) and math.isfinite(msg.epv):
            eh, ev = float(msg.eph) ** 2, float(msg.epv) ** 2
            out.position_covariance = [eh, 0.0, 0.0, 0.0, eh, 0.0, 0.0, 0.0, ev]
            out.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        else:
            out.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        self.pub_gps.publish(out)

    def _on_local(self, msg: VehicleLocalPosition):
        self._local = msg
        now = self.get_clock().now()
        if now.nanoseconds - self._last_pose_ns < self.pose_min_period_ns:
            return
        self._last_pose_ns = now.nanoseconds
        self._publish_agl(msg, now)
        if not (msg.xy_valid and msg.z_valid):
            return

        out = PoseStamped()
        out.header.stamp = now.to_msg()
        out.header.frame_id = self.odom_frame
        # NED -> ENU: x_enu = Este, y_enu = Norte, z_enu = -Down.
        out.pose.position.x = float(msg.y)
        out.pose.position.y = float(msg.x)
        out.pose.position.z = -float(msg.z)
        heading = float(msg.heading) if math.isfinite(msg.heading) else 0.0
        q = ned_heading_to_enu_quat(heading)
        o = out.pose.orientation
        o.x, o.y, o.z, o.w = q
        self.pub_pose.publish(out)

    def _publish_agl(self, msg: VehicleLocalPosition, now):
        # Solo si el EKF la estima CON un lidar: sin sensor de rango,
        # dist_bottom es un supuesto de terreno plano, no una medición.
        if not (msg.dist_bottom_valid and math.isfinite(msg.dist_bottom)
                and msg.dist_bottom_sensor_bitfield & VehicleLocalPosition.DIST_BOTTOM_SENSOR_RANGE):
            return
        out = Range()
        out.header.stamp = now.to_msg()
        out.header.frame_id = self.base_frame
        out.radiation_type = Range.INFRARED
        out.min_range = 0.0
        out.max_range = float('inf')
        out.range = float(msg.dist_bottom)
        self.pub_agl.publish(out)

    def _on_attitude(self, msg: VehicleAttitude):
        self._att = msg
        now = self.get_clock().now()
        if now.nanoseconds - self._last_att_ns < self.pose_min_period_ns:
            return
        self._last_att_ns = now.nanoseconds
        out = QuaternionStamped()
        out.header.stamp = now.to_msg()
        out.header.frame_id = 'ned'
        q = out.quaternion
        q.w, q.x, q.y, q.z = (float(v) for v in msg.q)     # PX4: [w, x, y, z]
        self.pub_att.publish(out)

    def _on_status(self, msg: VehicleStatus):
        self._vstatus = msg

    # ------------------------------------------------------------- periódicos
    def _publish_status(self):
        st = {}
        if self._vstatus is not None:
            v = self._vstatus
            st['armed'] = v.arming_state == VehicleStatus.ARMING_STATE_ARMED
            st['mode'] = NAV_STATES.get(v.nav_state, str(v.nav_state))
        if self._gps is not None:
            g = self._gps
            st['fix'] = FIX_NAMES.get(g.fix_type, str(g.fix_type))
            st['satellites'] = int(g.satellites_used)
            st['lat'] = round(g.latitude_deg, 7)
            st['lon'] = round(g.longitude_deg, 7)
            st['alt_msl'] = round(g.altitude_msl_m, 1)
            st['eph_m'] = round(float(g.eph), 2)
            st['ground_speed'] = round(float(g.vel_m_s), 2)
        if self._local is not None and self._local.z_valid:
            st['alt_rel'] = round(-float(self._local.z), 2)
            if math.isfinite(self._local.heading):
                st['heading_deg'] = round(math.degrees(self._local.heading) % 360.0, 1)
        if not st:
            return
        self.pub_status.publish(String(data=json.dumps(st)))

    def _check_alive(self):
        """Avisa (una vez por tópico) si no llega nada de PX4, con la causa
        probable: otra versión del mensaje (_vN) o px4_msgs distinto al firmware."""
        topics = None
        for key, got in (('gps_topic', self._gps), ('local_position_topic', self._local),
                         ('status_topic', self._vstatus), ('attitude_topic', self._att)):
            if got is not None or key in self._warned:
                continue
            self._warned.add(key)
            name = self.get_parameter(key).value
            if topics is None:
                # Solo tópicos que alguien PUBLICA: los que nosotros mismos
                # suscribimos también aparecen en el grafo.
                topics = [n for n, _ in self.get_topic_names_and_types()
                          if self.count_publishers(n) > 0]
            base = re.sub(r'_v\d+$', '', name)
            alt = [n for n in topics
                   if n != name and re.fullmatch(re.escape(base) + r'(_v\d+)?', n)]
            if alt:
                hint = (f'Existe {", ".join(alt)}: tu PX4 usa otra versión del mensaje. '
                        f'Poné {key}: {alt[0]} en el YAML.')
            elif name in topics:
                hint = ('El tópico existe pero no llega nada: px4_msgs no coincide con el '
                        'firmware (scripts/vendor_px4_msgs.sh, ver README).')
            else:
                hint = '¿Corre el MicroXRCEAgent? Revisá `ros2 topic list | grep fmu`.'
            self.get_logger().warn(f'No llega nada de PX4 en {name}. {hint}')


def main(args=None):
    run(GpsNode, args)


if __name__ == '__main__':
    main()
