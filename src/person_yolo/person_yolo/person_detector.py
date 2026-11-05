#!/usr/bin/env python3
"""Detección de personas lejanas y cercanas con YOLO, etiquetada con GPS.

Por frame:
  1. Inferencia sobre el frame completo (achicado a `imgsz`) -> personas cercanas.
  2. Si `tiling.enabled`: inferencia en batch sobre tiles a resolución nativa
     -> personas lejanas (ver tiling.py). Se fusiona todo con NMS.

Publica:
  /yolo/detections                  vision_msgs/Detection2DArray
  /yolo/person_count                std_msgs/Int32         <- Plot en Foxglove
                                    (personas en ESTE frame)
  /yolo/person_total                std_msgs/Int32 (latched): personas DISTINTAS
                                    marcadas desde el arranque (ver persons.py)
  /yolo/person_alert                std_msgs/String (JSON): UNA alerta por cada
                                    persona nueva = waypoint, tras seguirla
                                    counting.min_seen_sec y si no es una ya
                                    marcada que volvió a entrar al cuadro (ver
                                    persons.py): t, id, total, score, bbox,
                                    person_lat, person_lon, person_error_m,
                                    person_dist_m, agl_m, agl_source, geo (ver
                                    geolocate.py) + posición del dron: lat, lon,
                                    alt_msl, alt_rel, heading_deg, x, y, z
  /yolo/person_fix                  std_msgs/String (JSON): mismo formato, cada vez
                                    que la posición de una persona mejora mientras
                                    se la sigue viendo. `t` es el de su alerta
                                    (identifica a la persona); `fix_t`, el de ahora.
  /yolo/person_events               std_msgs/String (JSON, solo si hay personas):
                                    t, count, total, ids, max_score, lat, lon,
                                    alt_msl, alt_rel, heading_deg, x, y, z
  /yolo/image_annotated/compressed  sensor_msgs/CompressedImage (con HUD; solo se
                                    dibuja y codifica si alguien está suscripto)
Servicio:
  /yolo/person_total/reset          std_srvs/Trigger: vuelve el total a 0.

Base: uav_perception/yolo_detector.py de UAV_ros2-beta. Agnóstico a la fuente:
sirve con AirSim (/camera), con camera_node en la Jetson o con un rosbag.
"""
import csv
import json
import math
import os
import time

import numpy as np
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, QuaternionStamped
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.exceptions import ParameterUninitializedException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, NavSatFix, NavSatStatus, Range
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

import cv2

from person_yolo.geolocate import Camera, cos_tilt, locate, touches_edge
from person_yolo.model_path import is_downloadable, resolve_model_path
from person_yolo.node_main import run
from person_yolo.overlay import draw_boxes, draw_text_panel, enu_quat_to_heading_deg
from person_yolo.persons import Observation, PersonRegistry
from person_yolo.tiling import make_tiles, merge
from person_yolo.tracker import PersonCounter


def _boxes_np(results):
    """Results de ultralytics -> ndarray (N, 6) x1,y1,x2,y2,score,cls."""
    b = results.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 6))
    return np.concatenate([
        b.xyxy.cpu().numpy(),
        b.conf.cpu().numpy().reshape(-1, 1),
        b.cls.cpu().numpy().reshape(-1, 1),
    ], axis=1).astype(np.float64)


class PersonDetector(Node):

    def __init__(self):
        super().__init__('person_detector')

        # --- Parámetros -----------------------------------------------------
        self.declare_parameter('image_topic', '/camera')
        self.declare_parameter('model', 'yolo11n.pt')
        self.declare_parameter('device', 'auto')            # auto | cpu | cuda:0
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('conf', 0.35)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('half', True)                # fp16 en GPU (.pt)
        # Tipo explícito: una lista vacía sola se infiere como BYTE_ARRAY.
        self.declare_parameter(
            'classes', [],                                  # ids; vacío = por nombre
            ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER_ARRAY))
        self.declare_parameter(
            'person_class_names', ['person', 'pedestrian', 'people'],
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY))
        self.declare_parameter('max_rate_hz', 10.0)         # 0 = todos los frames

        self.declare_parameter('tiling.enabled', True)
        self.declare_parameter('tiling.tile_size', 640)
        self.declare_parameter('tiling.overlap', 0.2)
        self.declare_parameter('tiling.conf', 0.25)
        self.declare_parameter('tiling.every_n_frames', 1)
        self.declare_parameter('tiling.batch_size', 0)      # 0 = todos los tiles juntos
        self.declare_parameter('tiling.merge_thr', 0.5)
        self.declare_parameter('tiling.merge_metric', 'ios')  # iou | ios

        # Conteo de personas únicas (tracker.py)
        self.declare_parameter('counting.min_hits', 3)
        self.declare_parameter('counting.max_age_sec', 1.5)
        self.declare_parameter('counting.max_missed', 3)
        self.declare_parameter('counting.iou_thr', 0.2)
        self.declare_parameter('counting.dist_thr', 1.0)
        # Waypoint de cada persona (persons.py)
        self.declare_parameter('counting.min_seen_sec', 2.0)    # seguirla esto antes del waypoint
        self.declare_parameter('counting.max_wait_sec', 6.0)    # ...y hasta esto si no se ubica
        self.declare_parameter('counting.dedupe_radius_m', 4.0)  # 0 = sin buscar duplicados

        self.declare_parameter('annotated_scale', 1.0)
        self.declare_parameter('jpeg_quality', 75)
        self.declare_parameter('draw_tiles', False)
        self.declare_parameter('events_csv', '')            # '' = no guardar
        self.declare_parameter('log_period_sec', 5.0)
        self.declare_parameter('stale_sec', 2.0)            # GPS/pose más viejos = sin GPS

        # Ubicación de cada persona en el suelo (geolocate.py)
        self.declare_parameter('geolocate.enabled', True)
        self.declare_parameter('geolocate.camera_hfov_deg', 90.0)
        self.declare_parameter('geolocate.camera_pitch_deg', -30.0)  # - = hacia abajo
        self.declare_parameter('geolocate.person_height_m', 0.9)     # centro de la persona
        self.declare_parameter('geolocate.max_incidence_deg', 65.0)
        self.declare_parameter('geolocate.agl_topic', '/drone/agl')  # lidar (Range)
        self.declare_parameter('geolocate.agl_is_slant', False)      # lidar crudo, fijo al cuerpo
        self.declare_parameter('geolocate.sigma_pix', 2.0)
        self.declare_parameter('geolocate.sigma_att_deg', 1.0)
        self.declare_parameter('geolocate.sigma_alt_lidar', 0.3)
        self.declare_parameter('geolocate.sigma_alt_home', 3.0)      # sin lidar: suelo plano
        # Piso del error del GPS: PX4 puede informar 0.1 m (siempre en SITL) y el
        # margen de la persona saldría irrealmente chico.
        self.declare_parameter('geolocate.sigma_gps_min', 1.0)

        gp = self.get_parameter
        self.image_topic = gp('image_topic').value
        self.conf = float(gp('conf').value)
        self.iou = float(gp('iou').value)
        self.imgsz = int(gp('imgsz').value)
        self.tiling = bool(gp('tiling.enabled').value)
        self.tile_size = int(gp('tiling.tile_size').value)
        self.tile_overlap = float(gp('tiling.overlap').value)
        self.tile_conf = float(gp('tiling.conf').value)
        self.tile_every = max(1, int(gp('tiling.every_n_frames').value))
        self.tile_batch = int(gp('tiling.batch_size').value)
        self.merge_thr = float(gp('tiling.merge_thr').value)
        self.merge_metric = gp('tiling.merge_metric').value
        self.annotated_scale = float(gp('annotated_scale').value)
        self.jpeg_quality = int(gp('jpeg_quality').value)
        self.draw_tiles = bool(gp('draw_tiles').value)
        self.stale_sec = float(gp('stale_sec').value)
        self.geo = bool(gp('geolocate.enabled').value)
        self.geo_cam = Camera(float(gp('geolocate.camera_hfov_deg').value),
                              float(gp('geolocate.camera_pitch_deg').value))
        self.geo_kw = {
            'person_height_m': float(gp('geolocate.person_height_m').value),
            'max_incidence_deg': float(gp('geolocate.max_incidence_deg').value),
            'sigma_pix': float(gp('geolocate.sigma_pix').value),
            'sigma_att_deg': float(gp('geolocate.sigma_att_deg').value)}
        self.agl_is_slant = bool(gp('geolocate.agl_is_slant').value)
        self.sigma_alt_lidar = float(gp('geolocate.sigma_alt_lidar').value)
        self.sigma_alt_home = float(gp('geolocate.sigma_alt_home').value)
        self.sigma_gps_min = float(gp('geolocate.sigma_gps_min').value)
        self.registry = PersonRegistry(
            min_seen_sec=float(gp('counting.min_seen_sec').value),
            max_wait_sec=float(gp('counting.max_wait_sec').value),
            forget_sec=float(gp('counting.max_age_sec').value),
            dedupe_radius_m=float(gp('counting.dedupe_radius_m').value))
        # Una persona lejana solo aparece en los frames con tiles: max_missed
        # cuenta frames CON tiles, si no se la olvidaría entre uno y otro.
        max_missed = max(1, int(gp('counting.max_missed').value))
        if self.tiling:
            max_missed *= self.tile_every
        self.counter = PersonCounter(
            min_hits=max(1, int(gp('counting.min_hits').value)),
            max_age_sec=float(gp('counting.max_age_sec').value),
            max_missed=max_missed,
            iou_thr=float(gp('counting.iou_thr').value),
            dist_thr=float(gp('counting.dist_thr').value))

        # El 0.9 evita aliasing: con la cámara a ~10 Hz y período mínimo de
        # exactamente 100 ms, el jitter descarta un frame de cada dos.
        max_rate = float(gp('max_rate_hz').value)
        self.min_period = 0.9 / max_rate if max_rate > 0.0 else 0.0
        self.log_period = float(gp('log_period_sec').value)

        # --- Modelo ---------------------------------------------------------
        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().fatal(
                'Falta ultralytics. Instalalo con:\n'
                '  pip install --break-system-packages ultralytics\n'
                'En la Jetson ver README.md (torch de NVIDIA primero).')
            raise

        requested = gp('model').value
        model_path = resolve_model_path(requested, self._model_roots())
        if model_path is None:
            if not is_downloadable(requested):
                self.get_logger().fatal(
                    f'No existe el modelo "{requested}" (busqué en el directorio actual, '
                    '$YOLO_WS y en la raíz del workspace, también dentro de models/). '
                    'Un .engine se exporta en la Jetson con scripts/export_trt.sh.')
                raise FileNotFoundError(requested)
            model_path = requested              # nombre oficial: ultralytics lo descarga
        self.is_engine = model_path.endswith('.engine')
        self.device = self._resolve_device(gp('device').value)
        self.quant_kwargs = ({} if self.is_engine else self._resolve_quantization(
            bool(gp('half').value) and self.device != 'cpu'))

        self.get_logger().info(f'Cargando modelo "{model_path}" en {self.device}...')
        self.model = YOLO(model_path, task='detect')
        names = self.model.names
        self.names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else names
        self.classes = self._resolve_classes()
        if self.is_engine:
            self.get_logger().info(
                'Modelo TensorRT: imgsz y batch tienen que coincidir con el export '
                '(scripts/export_trt.sh). Si exportaste con batch=1, poné '
                'tiling.batch_size: 1.')

        # --- Pub / Sub ------------------------------------------------------
        # BEST_EFFORT + depth 1: si la inferencia tarda más que la cámara,
        # mejor perder frames viejos que acumular lag.
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.bridge = CvBridge()
        self.pub_det = self.create_publisher(Detection2DArray, '/yolo/detections', 10)
        self.pub_count = self.create_publisher(Int32, '/yolo/person_count', 10)
        self.pub_events = self.create_publisher(String, '/yolo/person_events', 50)
        # Latched: Foxglove abierto a mitad del vuelo ve el total enseguida.
        self.pub_total = self.create_publisher(
            Int32, '/yolo/person_total',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.pub_total.publish(Int32(data=0))
        self.pub_alert = self.create_publisher(String, '/yolo/person_alert', 50)
        self.pub_fix = self.create_publisher(String, '/yolo/person_fix', 50)
        self.create_service(Trigger, '/yolo/person_total/reset', self._on_reset_total)
        self.pub_jpg = self.create_publisher(
            CompressedImage, '/yolo/image_annotated/compressed', sensor_qos)

        self._frame_t = time.monotonic()    # llegada del último frame
        self.create_subscription(Image, self.image_topic, self.image_callback, sensor_qos)
        self.create_timer(5.0, self._check_camera)
        self.create_subscription(NavSatFix, '/drone/gps', self._on_gps, 10)
        self.create_subscription(PoseStamped, '/drone/pose', self._on_pose, 10)
        self.create_subscription(QuaternionStamped, '/drone/attitude', self._on_attitude, 10)
        self.create_subscription(Range, gp('geolocate.agl_topic').value, self._on_agl,
                                 sensor_qos)
        self.gps = None
        self.pose = None
        self.att = None
        self.agl = None
        self._gps_t = self._pose_t = self._att_t = self._agl_t = 0.0

        self.events_csv = None
        csv_path = gp('events_csv').value
        if csv_path:
            try:
                self._open_events_csv(os.path.expanduser(csv_path))
            except OSError as exc:       # el CSV es opcional: no vale perder la detección
                self.get_logger().error(f'Sin CSV de eventos, no se pudo abrir: {exc}')

        # --- Estadísticas ---------------------------------------------------
        self._frame_idx = 0
        self._last_infer = 0.0
        self._frames_in = self._frames_done = self._frames_tiled = 0
        self._t_full = self._t_tiles = 0.0
        self._last_log = time.monotonic()
        self._last_ms = 0.0
        self._warned_tiles = False

        self.get_logger().info(
            f'person_detector listo (tiling {"ON" if self.tiling else "OFF"}), '
            f'esperando imágenes en {self.image_topic}...')

    # ------------------------------------------------------------------ setup
    def _model_roots(self):
        """Dónde buscar `model:` (ver model_path.py)."""
        roots = [os.getcwd()]
        if os.environ.get('YOLO_WS'):
            roots.append(os.environ['YOLO_WS'])
        try:
            # <ws>/install/person_yolo/share/person_yolo -> <ws>
            share = get_package_share_directory('person_yolo')
            roots.append(os.path.abspath(os.path.join(share, '..', '..', '..', '..')))
        except PackageNotFoundError:
            pass
        return roots

    def _resolve_classes(self):
        """Ids de clase a detectar. `classes` explícito gana; si no, se buscan
        por nombre (así un peso VisDrone con 'pedestrian' funciona sin tocar
        nada). Si no hay coincidencias, se avisa y se detecta todo."""
        try:
            value = self.get_parameter('classes').value
        except ParameterUninitializedException:
            value = None
        if value:
            ids = [int(c) for c in value]
        else:
            wanted = {n.lower() for n in self.get_parameter('person_class_names').value}
            ids = [i for i, n in self.names.items() if str(n).lower() in wanted]
        if not ids:
            self.get_logger().warn(
                f'El modelo no tiene ninguna clase de persona {sorted(self.names.values())}; '
                'se detectan TODAS las clases. Ajustá person_class_names o classes.')
            return None
        self.get_logger().info(
            'Clases detectadas: ' + ', '.join(f'{i}={self.names.get(i)}' for i in ids))
        return ids

    @staticmethod
    def _resolve_quantization(want_fp16: bool) -> dict:
        """Hasta ultralytics 8.3 era `half=True`; desde 8.4 es `quantize='fp16'`
        y `half` imprime un warning por frame."""
        if not want_fp16:
            return {}
        from ultralytics.utils import DEFAULT_CFG_DICT
        return {'quantize': 'fp16'} if 'quantize' in DEFAULT_CFG_DICT else {'half': True}

    def _resolve_device(self, requested: str) -> str:
        if requested != 'auto':
            return requested
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda:0'
        except ImportError:
            pass
        self.get_logger().warn('Sin GPU para torch: se corre en CPU (mucho más lento). '
                               'Considerá tiling.enabled: false o every_n_frames > 1.')
        return 'cpu'

    def _open_events_csv(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        new = not os.path.exists(path)
        self.events_csv = open(path, 'a', newline='', encoding='utf-8')
        self._csv = csv.writer(self.events_csv)
        if new:
            self._csv.writerow(['t', 'count', 'max_score', 'lat', 'lon', 'alt_msl',
                                'alt_rel', 'heading_deg', 'x', 'y', 'z'])
        self.get_logger().info(f'Eventos de personas -> {path}')

    # ------------------------------------------------------------------ GPS
    def _on_gps(self, msg: NavSatFix):
        valid = (msg.status.status >= NavSatStatus.STATUS_FIX
                 and math.isfinite(msg.latitude) and math.isfinite(msg.longitude))
        self.gps = msg if valid else None
        self._gps_t = time.monotonic()

    def _on_pose(self, msg: PoseStamped):
        self.pose = msg
        self._pose_t = time.monotonic()

    def _on_attitude(self, msg: QuaternionStamped):
        q = msg.quaternion
        self.att = (q.w, q.x, q.y, q.z)
        self._att_t = time.monotonic()

    def _on_agl(self, msg: Range):
        self.agl = msg
        self._agl_t = time.monotonic()

    def _fresh(self, t):
        # Respecto de la llegada del frame, no de ahora: durante la inferencia no
        # corren otros callbacks, y si tarda más que stale_sec (CPU) todo parecería
        # viejo aunque esté llegando bien.
        return self._frame_t - t <= self.stale_sec

    def _check_camera(self):
        """Sin frames no corre image_callback ni sus logs: sin este aviso, si se
        cae la cámara o airsim_node se dejaría de detectar sin que nada lo diga."""
        idle = time.monotonic() - self._frame_t
        if idle > 5.0:
            self.get_logger().error(
                f'Sin imágenes en {self.image_topic} hace {idle:.0f} s: NO se están '
                'detectando personas. ¿Se cayó la cámara o airsim_node?')

    def _height(self):
        """(altura de la cámara sobre el suelo, fuente, sigma) o None.
        Lidar si hay una lectura fresca y en rango; si no, altura sobre el
        despegue (supone suelo plano, más error)."""
        r = self.agl
        if (r is not None and self._fresh(self._agl_t) and math.isfinite(r.range)
                and r.min_range <= r.range < r.max_range):
            h = r.range
            if self.agl_is_slant:               # lidar fijo al cuerpo: medida inclinada
                h *= max(0.0, cos_tilt(self.att))
            return h, 'lidar', self.sigma_alt_lidar
        if self.pose is not None and self._fresh(self._pose_t):
            return self.pose.pose.position.z, 'home', self.sigma_alt_home
        return None

    def _locate_person(self, box, width, height):
        """Posición de la persona en el suelo (dict de geolocate.locate + agl)
        y un motivo si no se pudo."""
        if not self.geo:
            return None, 'desactivado'
        if self.gps is None or not self._fresh(self._gps_t):
            return None, 'sin GPS'
        if self.att is None or not self._fresh(self._att_t):
            return None, 'sin actitud'
        hgt = self._height()
        if hgt is None:
            return None, 'sin altura'
        cov = self.gps.position_covariance
        sigma_gps = max(self.sigma_gps_min, math.sqrt(cov[0])
                        if self.gps.position_covariance_type and cov[0] > 0 else 2.5)
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        r = locate((x1 + x2) / 2.0, (y1 + y2) / 2.0, width, height, self.geo_cam,
                   self.att, hgt[0], self.gps.latitude, self.gps.longitude,
                   sigma_alt=hgt[2], sigma_gps=sigma_gps, **self.geo_kw)
        if r is None:
            return None, 'rayo muy rasante o sobre el horizonte'
        r.update(agl_m=hgt[0], agl_source=hgt[1])
        return r, 'ok'

    def _where(self):
        """Posición actual del dron para etiquetar detecciones (None si falta o
        es más vieja que stale_sec: si se corta la Pixhawk, una alerta con la
        posición de hace minutos sería engañosa)."""
        w = {'lat': None, 'lon': None, 'alt_msl': None, 'alt_rel': None,
             'heading_deg': None, 'x': None, 'y': None, 'z': None}
        if self.gps is not None and self._fresh(self._gps_t):
            w.update(lat=round(self.gps.latitude, 8), lon=round(self.gps.longitude, 8),
                     alt_msl=round(self.gps.altitude, 2))
        if self.pose is not None and self._fresh(self._pose_t):
            p, q = self.pose.pose.position, self.pose.pose.orientation
            w.update(x=round(p.x, 2), y=round(p.y, 2), z=round(p.z, 2),
                     alt_rel=round(p.z, 2),
                     heading_deg=round(enu_quat_to_heading_deg(q.z, q.w), 1))
        return w

    # --------------------------------------------------------------- inferencia
    def _predict(self, images, imgsz, conf):
        return self.model.predict(images, imgsz=imgsz, conf=conf, iou=self.iou,
                                  classes=self.classes, device=self.device,
                                  verbose=False, **self.quant_kwargs)

    def _infer(self, frame):
        h, w = frame.shape[:2]
        t0 = time.monotonic()
        full = _boxes_np(self._predict(frame, self.imgsz, self.conf)[0])
        t1 = time.monotonic()
        self._t_full += t1 - t0

        tiles = []
        tile_results = []
        if self.tiling and self._frame_idx % self.tile_every == 0:
            tiles = make_tiles(w, h, self.tile_size, self.tile_overlap)
            if len(tiles) <= 1:
                if not self._warned_tiles:
                    self._warned_tiles = True
                    self.get_logger().warn(
                        f'Imagen {w}x{h} <= tile {self.tile_size}: el tiling no aporta '
                        'nada. Subí la resolución de la cámara (en AirSim: settings.json '
                        'Width/Height) o bajá tiling.tile_size.')
                tiles = []
            else:
                crops = [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in tiles]
                bs = self.tile_batch if self.tile_batch > 0 else len(crops)
                for i in range(0, len(crops), bs):
                    res = self._predict(crops[i:i + bs], self.tile_size, self.tile_conf)
                    for rect, r in zip(tiles[i:i + bs], res):
                        tile_results.append((rect, _boxes_np(r)))
                self._t_tiles += time.monotonic() - t1
                self._frames_tiled += 1

        boxes, src = merge(full, tile_results, w, h, self.merge_thr, self.merge_metric)
        # Las de tile pasaron con tiling.conf (más bajo); las del frame completo ya
        # vienen filtradas por conf. Se conserva todo lo que sobrevivió al NMS.
        self._last_ms = 1000.0 * (time.monotonic() - t0)
        return boxes, src, tiles

    # --------------------------------------------------------------- callback
    def image_callback(self, msg: Image):
        self._frames_in += 1
        now = self._frame_t = time.monotonic()
        if self.min_period and (now - self._last_infer) < self.min_period:
            return
        self._last_infer = now

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'No se pudo convertir la imagen: {exc}',
                                    throttle_duration_sec=5.0)
            return

        boxes, src, tiles = self._infer(frame)
        self._frame_idx += 1
        self._frames_done += 1

        # Reloj monotónico y no el stamp: con un rosbag en loop el stamp va
        # para atrás y los tracks no expirarían nunca.
        tracks = self.counter.update(boxes, time.monotonic())

        where = self._where()
        stamp = msg.header.stamp
        stamp_t = round(stamp.sec + stamp.nanosec * 1e-9, 3)
        obs = self._observations(boxes, tracks, where, frame.shape[1], frame.shape[0])
        self._publish_person_events(self.registry.update(time.monotonic(), stamp_t, obs),
                                    stamp_t)
        # Id de persona (el del waypoint) por caja; 0 mientras no lo es.
        ids = np.array([self.registry.person_id(t) for t in tracks], dtype=np.int64)
        self._publish_detections(msg, boxes, ids)
        self.pub_count.publish(Int32(data=len(boxes)))
        if len(boxes):
            self._publish_event(msg, boxes, ids, where)
        if self.pub_jpg.get_subscription_count():       # nadie mirando: no gastar CPU
            self._publish_annotated(msg, frame, boxes, src, ids, tiles, where)
        self._log_stats()

    def _on_reset_total(self, request, response):
        self.counter.reset()
        self.registry.reset()
        self.pub_total.publish(Int32(data=0))
        response.success = True
        response.message = 'total de personas en 0'
        self.get_logger().info('/yolo/person_total/reset: ' + response.message)
        return response

    # ------------------------------------------------------------ publicación
    def _publish_detections(self, src_msg: Image, boxes, ids):
        out = Detection2DArray()
        out.header = src_msg.header
        for (x1, y1, x2, y2, score, cls), pid in zip(boxes, ids):
            label = str(self.names.get(int(cls), int(cls)))
            det = Detection2D()
            det.header = src_msg.header
            det.id = str(pid) if pid else ''       # id de tracking ('' = sin confirmar)
            bbox = BoundingBox2D()
            bbox.center.position.x = (x1 + x2) / 2.0
            bbox.center.position.y = (y1 + y2) / 2.0
            bbox.size_x = float(x2 - x1)
            bbox.size_y = float(y2 - y1)
            det.bbox = bbox
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score = float(score)
            det.results.append(hyp)
            out.detections.append(det)
        self.pub_det.publish(out)

    def _publish_event(self, src_msg: Image, boxes, ids, where):
        stamp = src_msg.header.stamp
        ev = {'t': round(stamp.sec + stamp.nanosec * 1e-9, 3),
              'count': int(len(boxes)),
              'total': self.registry.total,
              'ids': [int(i) for i in ids],
              'max_score': round(float(boxes[:, 4].max()), 3),
              'scores': [round(float(s), 3) for s in boxes[:, 4]],
              **where}
        self.pub_events.publish(String(data=json.dumps(ev)))
        if self.events_csv is not None:
            self._csv.writerow([ev['t'], ev['count'], ev['max_score'], where['lat'],
                                where['lon'], where['alt_msl'], where['alt_rel'],
                                where['heading_deg'], where['x'], where['y'], where['z']])
            self.events_csv.flush()

    def _person_msg(self, t, pid, box, geo, why, where):
        x1, y1, x2, y2, score = (float(v) for v in box[:5])
        return {'t': t,
                'id': int(pid),
                'total': self.registry.total,
                'score': round(score, 3),
                'bbox': [round(x1), round(y1), round(x2), round(y2)],
                'person_lat': round(geo['lat'], 8) if geo else None,
                'person_lon': round(geo['lon'], 8) if geo else None,
                'person_error_m': round(geo['error_m'], 1) if geo else None,
                'person_dist_m': round(geo['dist_m'], 1) if geo else None,
                'agl_m': round(geo['agl_m'], 2) if geo else None,
                'agl_source': geo['agl_source'] if geo else None,
                'geo': why,
                **where}

    def _observations(self, boxes, tracks, where, width, height):
        """Una Observation por track confirmado, con la posición de la persona si
        se pudo calcular. Una caja cortada por el borde no se ubica: su centro
        no es el de la persona."""
        out = []
        for box, tid in zip(boxes, tracks):
            if not tid:
                continue
            if touches_edge(box, width, height):
                geo, why = None, 'caja cortada por el borde'
            else:
                geo, why = self._locate_person(box, width, height)
            payload = {'box': box, 'geo': geo, 'why': why, 'where': where}
            if geo:
                out.append(Observation(int(tid), geo['lat'], geo['lon'], geo['error_m'],
                                       payload))
            else:
                out.append(Observation(int(tid), payload=payload))
        return out

    def _publish_person_events(self, events, stamp_t):
        """Alerta = waypoint nuevo (la persona se siguió counting.min_seen_sec, ver
        persons.py); fix = mejor posición de una persona ya marcada. Llevan la
        posición de la PERSONA en el suelo si se pudo calcular, y siempre la del
        dron en el momento de esa observación."""
        if any(e.kind == 'alert' for e in events):
            self.pub_total.publish(Int32(data=self.registry.total))
        for e in events:
            p = e.obs.payload
            if e.kind == 'duplicate':
                self.get_logger().info(
                    f'Persona #{e.person_id} vuelve a estar a la vista: no es nueva')
                continue
            m = self._person_msg(e.t, e.person_id, p['box'], p['geo'], p['why'], p['where'])
            if e.kind == 'fix':
                m['fix_t'] = stamp_t
                self.pub_fix.publish(String(data=json.dumps(m)))
                continue
            self.pub_alert.publish(String(data=json.dumps(m)))
            geo, why, where, pid = p['geo'], p['why'], p['where'], e.person_id
            score = float(p['box'][4])
            if geo:
                pos = (f'lat={geo["lat"]:.6f} lon={geo["lon"]:.6f} '
                       f'(±{geo["error_m"]:.0f} m, a {geo["dist_m"]:.0f} m del dron, '
                       f'altura por {geo["agl_source"]})')
            elif where['lat'] is not None:
                pos = f'posición del dron lat={where["lat"]:.6f} lon={where["lon"]:.6f} ({why})'
            else:
                pos = 'sin GPS'
            self.get_logger().warn(
                f'ALERTA: persona nueva #{pid} (score {score:.2f}, total {self.registry.total})'
                f' en {pos}')

    def _publish_annotated(self, src_msg, frame, boxes, src, ids, tiles, where):
        img = frame.copy()
        if self.draw_tiles:
            for x0, y0, x1, y1 in tiles:
                cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (90, 90, 90), 1)
        draw_boxes(img, boxes, src, ids=ids)

        if self.annotated_scale != 1.0:
            img = cv2.resize(img, None, fx=self.annotated_scale, fy=self.annotated_scale,
                             interpolation=cv2.INTER_AREA)
        self._draw_hud(img, len(boxes), bool(tiles), where)

        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY),
                                             self.jpeg_quality])
        if ok:
            jpg = CompressedImage()
            jpg.header = src_msg.header
            jpg.format = 'jpeg'
            jpg.data = buf.tobytes()
            self.pub_jpg.publish(jpg)

    def _draw_hud(self, img, count, tiled, where):
        if where['lat'] is not None:
            pos = f'{where["lat"]:.6f}, {where["lon"]:.6f}'
        else:
            pos = 'sin GPS'
        alt = f'{where["alt_rel"]:.1f} m' if where['alt_rel'] is not None else '-'
        hdg = f'{where["heading_deg"]:.0f}' if where['heading_deg'] is not None else '-'
        lines = [
            f'PERSONAS: {count}  TOTAL: {self.registry.total}',
            f'GPS {pos}  ALT {alt}  RUMBO {hdg}',
            f'{self._last_ms:.0f} ms  tiling {"ON" if tiled else "off"}  ({self.device})',
        ]
        draw_text_panel(img, lines, highlight_first=bool(count))

    def _log_stats(self):
        if not self.log_period:
            return
        now = time.monotonic()
        elapsed = now - self._last_log
        if elapsed < self.log_period:
            return
        n = max(self._frames_done, 1)
        tiles_ms = 1000.0 * self._t_tiles / max(self._frames_tiled, 1)
        self.get_logger().info(
            f'cámara {self._frames_in / elapsed:.1f} Hz | '
            f'inferencia {self._frames_done / elapsed:.1f} Hz | '
            f'full {1000.0 * self._t_full / n:.0f} ms + tiles {tiles_ms:.0f} ms '
            f'({self._frames_tiled}/{self._frames_done} frames) | {self.device} | '
            f'personas distintas {self.registry.total}'
            + ('' if self.gps and self._fresh(self._gps_t) else ' | SIN GPS'))
        self._frames_in = self._frames_done = self._frames_tiled = 0
        self._t_full = self._t_tiles = 0.0
        self._last_log = now

    def destroy_node(self):
        if self.events_csv is not None:
            self.events_csv.close()
        super().destroy_node()


def main(args=None):
    run(PersonDetector, args)


if __name__ == '__main__':
    main()
