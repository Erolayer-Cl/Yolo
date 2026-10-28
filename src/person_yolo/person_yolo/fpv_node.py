#!/usr/bin/env python3
"""Vista FPV en el navegador: video de la cámara a su fps real + HUD de piloto.

    http://localhost:8080/                 página a pantalla completa, botón / tecla Y
                                           para mostrar u ocultar las cajas de YOLO
    http://localhost:8080/stream?boxes=1   MJPEG con cajas  (sirve en VLC, otra pestaña)
    http://localhost:8080/stream?boxes=0   MJPEG sin cajas

A diferencia de /yolo/image_annotated/compressed (que sale solo cuando termina
una inferencia, ~5-10 Hz), acá se dibuja CADA frame de la cámara y encima las
últimas cajas de /yolo/detections. YOLO puede ir lento sin que el video se trabe.

HUD: modo, armado, altura, velocidad, rumbo, personas (en cuadro y total
distintas), fps y cruz central. Cada caja lleva el #id de persona del tracker.
Si deja de llegar la cámara, el video sigue con un aviso "SIN CAMARA".

Solo stdlib para el servidor HTTP: no hace falta instalar nada. En el dron real
se abre desde tierra en http://<ip-jetson>:8080 (sirve para mirar, no para
pilotear: el WiFi agrega 100-300 ms).
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from vision_msgs.msg import Detection2DArray

from person_yolo.node_main import run
from person_yolo.overlay import (
    COLOR_ALERT,
    draw_banner,
    draw_boxes,
    draw_crosshair,
    draw_text_panel,
    enu_quat_to_heading_deg,
)

BOUNDARY = 'fpvframe'

PAGE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FPV dron</title>
<style>
  html, body { margin: 0; height: 100%; background: #000; overflow: hidden; }
  img { width: 100vw; height: 100vh; object-fit: contain; display: block; }
  button { position: fixed; top: 12px; right: 12px; padding: 10px 16px;
           font: 600 15px system-ui, sans-serif; border: 0; border-radius: 6px;
           background: rgba(0,0,0,.65); color: #fff; cursor: pointer; }
  button.on { background: rgba(40,160,60,.85); }
  #help { position: fixed; bottom: 10px; right: 14px; color: #aaa;
          font: 13px system-ui, sans-serif; }
</style></head>
<body>
<img id="v" alt="video FPV">
<button id="b"></button>
<div id="help">Y = cajas YOLO &nbsp;·&nbsp; F = pantalla completa</div>
<script>
  const DEFAULT_BOXES = __BOXES_DEFAULT__;
  let boxes = DEFAULT_BOXES;
  try { const s = localStorage.getItem('fpv_boxes'); if (s !== null) boxes = s === '1'; } catch (e) {}
  const v = document.getElementById('v'), b = document.getElementById('b');
  function show() {
    v.src = '/stream?boxes=' + (boxes ? 1 : 0) + '&t=' + Date.now();
    b.textContent = 'Cajas YOLO: ' + (boxes ? 'ON' : 'OFF');
    b.className = boxes ? 'on' : '';
    try { localStorage.setItem('fpv_boxes', boxes ? '1' : '0'); } catch (e) {}
  }
  function toggle() { boxes = !boxes; show(); }
  b.onclick = toggle;
  document.addEventListener('keydown', e => {
    if (e.key === 'y' || e.key === 'Y') toggle();
    if (e.key === 'f' || e.key === 'F') {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen().catch(() => {});
    }
  });
  // Si se corta el stream (nodo reiniciado), reintenta solo.
  v.onerror = () => setTimeout(show, 1000);
  show();
</script>
</body></html>
"""


class StreamHub:
    """Último JPEG de cada vista (con / sin cajas), compartido entre clientes.

    Cada frame se codifica una vez por vista, y solo si alguien la está mirando.
    Un cliente lento no acumula frames: siempre recibe el más nuevo.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._frames = {True: (0, None), False: (0, None)}
        self._viewers = {True: 0, False: 0}
        self._stopped = False

    def viewers(self, boxes: bool) -> int:
        with self._cond:
            return self._viewers[boxes]

    def add_viewer(self, boxes: bool, delta: int):
        with self._cond:
            self._viewers[boxes] += delta

    def publish(self, boxes: bool, jpg: bytes):
        with self._cond:
            seq = self._frames[boxes][0] + 1
            self._frames[boxes] = (seq, jpg)
            self._cond.notify_all()

    def wait(self, boxes: bool, last_seq: int, timeout: float = 2.0):
        """(seq, jpg) del primer frame más nuevo que `last_seq`; None si se paró
        el nodo o pasó `timeout` sin frames."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._stopped or self._frames[boxes][0] > last_seq, timeout)
            if self._stopped:
                return None
            seq, jpg = self._frames[boxes]
            return (seq, jpg) if seq > last_seq and jpg is not None else None

    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    @property
    def stopped(self) -> bool:
        return self._stopped


def make_handler(hub: StreamHub, page: bytes, logger):

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, fmt, *args):     # sin una línea por request en el log
            pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ('/', '/index.html'):
                self._send(200, 'text/html; charset=utf-8', page)
            elif url.path == '/stream':
                q = parse_qs(url.query)
                self._stream(q.get('boxes', ['1'])[0] not in ('0', 'false', 'off'))
            else:
                self._send(404, 'text/plain', b'no existe\n')

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _stream(self, boxes):
            self.send_response(200)
            self.send_header('Content-Type', f'multipart/x-mixed-replace; boundary={BOUNDARY}')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()
            hub.add_viewer(boxes, +1)
            logger.info(f'Cliente FPV conectado ({"con" if boxes else "sin"} cajas) '
                        f'desde {self.client_address[0]}')
            last = 0
            try:
                while not hub.stopped:
                    got = hub.wait(boxes, last)
                    if got is None:
                        continue
                    last, jpg = got
                    self.wfile.write(
                        f'--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n'
                        f'Content-Length: {len(jpg)}\r\n\r\n'.encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b'\r\n')
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass                            # el navegador cerró o cambió de vista
            finally:
                hub.add_viewer(boxes, -1)

    return Handler


class FpvNode(Node):

    def __init__(self):
        super().__init__('fpv_node')

        self.declare_parameter('image_topic', '/camera')
        self.declare_parameter('detections_topic', '/yolo/detections')
        self.declare_parameter('port', 8080)
        self.declare_parameter('bind_address', '0.0.0.0')
        self.declare_parameter('scale', 1.0)                # achicar antes de codificar
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('max_fps', 30.0)             # 0 = todos los frames
        self.declare_parameter('boxes_default', True)       # vista inicial de la página
        self.declare_parameter('boxes_max_age_sec', 0.6)    # cajas más viejas se ocultan
        self.declare_parameter('camera_timeout_sec', 1.0)

        gp = self.get_parameter
        self.scale = float(gp('scale').value)
        self.jpeg_quality = int(gp('jpeg_quality').value)
        max_fps = float(gp('max_fps').value)
        self.min_period = 0.9 / max_fps if max_fps > 0.0 else 0.0
        self.boxes_max_age = float(gp('boxes_max_age_sec').value)
        self.camera_timeout = float(gp('camera_timeout_sec').value)

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.bridge = CvBridge()
        self.create_subscription(Image, gp('image_topic').value, self._on_image, sensor_qos)
        self.create_subscription(Detection2DArray, gp('detections_topic').value,
                                 self._on_detections, 10)
        self.create_subscription(PoseStamped, '/drone/pose', self._on_pose, 10)
        self.create_subscription(String, '/drone/status', self._on_status, 10)
        # Mismo QoS latched que el publicador: el total llega aunque el FPV
        # arranque después que el detector.
        self.create_subscription(
            Int32, '/yolo/person_total', self._on_total,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self.boxes = np.zeros((0, 5))
        self.ids = np.zeros(0, dtype=int)
        self.boxes_t = 0.0
        self.total = None
        self.pose = None
        self.status = {}
        self.last_frame = None              # último frame (ya escalado), para el aviso
        self.last_frame_t = 0.0
        self._last_sent = 0.0
        self._fps = 0.0
        self._fps_n = 0
        self._fps_t0 = time.monotonic()

        # --- Servidor HTTP en su propio hilo ----------------------------------
        self.hub = StreamHub()
        page = PAGE.replace('__BOXES_DEFAULT__',
                            'true' if gp('boxes_default').value else 'false').encode()
        port = int(gp('port').value)
        addr = gp('bind_address').value
        self.server = ThreadingHTTPServer((addr, port),
                                          make_handler(self.hub, page, self.get_logger()))
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, name='fpv_http',
                         daemon=True).start()

        self.create_timer(0.25, self._check_camera)
        self.get_logger().info(
            f'FPV en http://localhost:{port}/  (Y = mostrar/ocultar cajas YOLO)')

    # ------------------------------------------------------------- callbacks
    def _on_detections(self, msg: Detection2DArray):
        rows, ids = [], []
        for d in msg.detections:
            cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
            hw, hh = d.bbox.size_x / 2.0, d.bbox.size_y / 2.0
            score = d.results[0].hypothesis.score if d.results else 0.0
            rows.append((cx - hw, cy - hh, cx + hw, cy + hh, score))
            ids.append(int(d.id) if d.id.isdigit() else 0)     # '' = sin confirmar
        self.boxes = np.array(rows, dtype=np.float64).reshape(-1, 5)
        self.ids = np.array(ids, dtype=int)
        self.boxes_t = time.monotonic()

    def _on_total(self, msg: Int32):
        self.total = msg.data

    def _on_pose(self, msg: PoseStamped):
        self.pose = msg

    def _on_status(self, msg: String):
        try:
            self.status = json.loads(msg.data)
        except ValueError:
            pass

    def _on_image(self, msg: Image):
        now = time.monotonic()
        if self.min_period and (now - self._last_sent) < self.min_period:
            return
        if not (self.hub.viewers(True) or self.hub.viewers(False)):
            self.last_frame_t = now             # nadie mirando: no gastar CPU
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'No se pudo convertir la imagen: {exc}',
                                    throttle_duration_sec=5.0)
            return
        if self.scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA)
        self._last_sent = now
        self.last_frame = frame
        self.last_frame_t = now
        self._count_fps(now)
        self._render_and_publish(frame, camera_ok=True)

    def _check_camera(self):
        """Si se corta la cámara, los clientes siguen recibiendo un frame con aviso
        (si no, el navegador se queda congelado en la última imagen sin avisar)."""
        if time.monotonic() - self.last_frame_t < self.camera_timeout:
            return
        if not (self.hub.viewers(True) or self.hub.viewers(False)):
            return
        if self.last_frame is not None:
            base = (self.last_frame * 0.35).astype(np.uint8)
        else:
            base = np.zeros((360, 640, 3), dtype=np.uint8)
        self._fps = 0.0
        self._render_and_publish(base, camera_ok=False)

    # ------------------------------------------------------------- render
    def _count_fps(self, now):
        self._fps_n += 1
        if now - self._fps_t0 >= 1.0:
            self._fps = self._fps_n / (now - self._fps_t0)
            self._fps_n = 0
            self._fps_t0 = now

    def _fresh_boxes(self):
        if time.monotonic() - self.boxes_t > self.boxes_max_age:
            return np.zeros((0, 5)), np.zeros(0, dtype=int)
        return self.boxes, self.ids

    def _hud_lines(self, n_people):
        st = self.status
        mode = st.get('mode', '?')
        armed = 'ARMADO' if st.get('armed') else 'desarmado'
        if self.pose is not None:
            p, q = self.pose.pose.position, self.pose.pose.orientation
            alt = f'{p.z:.1f} m'
            hdg = f'{enu_quat_to_heading_deg(q.z, q.w):03.0f}'
        else:
            alt = f'{st["alt_rel"]:.1f} m' if 'alt_rel' in st else '-'
            hdg = f'{st["heading_deg"]:03.0f}' if 'heading_deg' in st else '-'
        vel = f'{st["ground_speed"]:.1f} m/s' if 'ground_speed' in st else '-'
        total = '-' if self.total is None else self.total
        return [
            f'{mode}  {armed}',
            f'ALT {alt}  VEL {vel}  RUMBO {hdg}',
            f'PERSONAS {n_people}  TOTAL {total}   {self._fps:.0f} fps',
        ]

    def _render_and_publish(self, frame, camera_ok):
        boxes, ids = self._fresh_boxes()
        n_people = len(boxes)
        for with_boxes in (True, False):
            if not self.hub.viewers(with_boxes):
                continue
            img = frame.copy()
            if with_boxes and camera_ok:
                draw_boxes(img, boxes, scale=self.scale, ids=ids)
            draw_crosshair(img)
            draw_text_panel(img, self._hud_lines(n_people),
                            highlight_first=bool(self.status.get('armed')), width=520)
            if not camera_ok:
                draw_banner(img, 'SIN CAMARA', COLOR_ALERT)
            ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY),
                                                 self.jpeg_quality])
            if ok:
                self.hub.publish(with_boxes, buf.tobytes())

    def destroy_node(self):
        self.hub.stop()
        self.server.shutdown()
        self.server.server_close()
        super().destroy_node()


def main(args=None):
    run(FpvNode, args)


if __name__ == '__main__':
    main()
