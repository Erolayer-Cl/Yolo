#!/usr/bin/env python3
"""Cámara USB / CSI -> /camera (sensor_msgs/Image). Solo para el dron real:
en SITL la imagen la publica airsim_node y este nodo no se lanza.

`source` puede ser:
  - un índice ("0") o un dispositivo ("/dev/video0")        -> V4L2 (USB)
  - un pipeline GStreamer completo (contiene "!")           -> CSI en la Jetson, p.ej.
    nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1920,height=1080,
    framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert !
    video/x-raw,format=BGR ! appsink drop=1 max-buffers=1

La captura corre en un hilo que guarda solo el ÚLTIMO frame: si el detector va
más lento que la cámara, se descartan frames viejos en vez de acumular lag.
"""
import threading
import time

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

import cv2

from person_yolo.node_main import run


def _has_gstreamer():
    for line in cv2.getBuildInformation().splitlines():
        if 'GStreamer' in line:
            return 'YES' in line
    return False


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')
        self.declare_parameter('source', '0')
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('fourcc', 'MJPG')     # USB: MJPG permite 1080p@30
        self.declare_parameter('frame_id', 'camera')
        self.declare_parameter('topic', '/camera')

        gp = self.get_parameter
        self.source = str(gp('source').value)
        self.width, self.height = int(gp('width').value), int(gp('height').value)
        self.fps = float(gp('fps').value)
        self.fourcc = str(gp('fourcc').value)
        self.frame_id = gp('frame_id').value

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(Image, gp('topic').value, qos)
        self.bridge = CvBridge()

        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._sent_seq = 0
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self.create_timer(1.0 / max(self.fps, 1.0), self._publish)

    def _open(self):
        if '!' in self.source:
            if not _has_gstreamer():
                # Pasa en la Jetson si `pip install ultralytics` trajo opencv-python
                # (sin GStreamer) y tapa al OpenCV de JetPack: el pipeline nunca abre.
                self.get_logger().error(
                    f'Este OpenCV ({cv2.__file__}) no tiene GStreamer: la cámara CSI no '
                    'puede abrir. Sacá el de pip: pip uninstall -y opencv-python '
                    'opencv-python-headless (ver README, sección Jetson).',
                    throttle_duration_sec=30.0)
                return None
            cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
        else:
            src = int(self.source) if self.source.isdigit() else self.source
            cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            if self.fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f'Cámara abierta: {self.source} ({w}x{h})')
        if '!' not in self.source and (w, h) != (self.width, self.height):
            self.get_logger().warn(f'Se pidió {self.width}x{self.height}; la cámara '
                                   f'entrega {w}x{h}.')
        return cap

    def _capture_loop(self):
        cap = None
        while self._running:
            if cap is None:
                cap = self._open()
                if cap is None:
                    self.get_logger().error(
                        f'No se pudo abrir la cámara "{self.source}", reintento en 2 s...',
                        throttle_duration_sec=10.0)
                    time.sleep(2.0)
                    continue
            ok, frame = cap.read()
            if not ok:
                self.get_logger().error('La cámara dejó de entregar frames; reabriendo...')
                cap.release()
                cap = None
                time.sleep(1.0)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1
        if cap is not None:
            cap.release()

    def _publish(self):
        with self._lock:
            if self._frame is None or self._seq == self._sent_seq:
                return
            frame, self._sent_seq = self._frame, self._seq
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

    def destroy_node(self):
        self._running = False
        self._thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    run(CameraNode, args)


if __name__ == '__main__':
    main()
