"""Dónde está el peso de YOLO que pide `model:` en el YAML.

Una ruta relativa se busca en cada raíz como `<raíz>/<ruta>` y
`<raíz>/models/<ruta>`. person_detector usa como raíces el directorio actual,
$YOLO_WS (lo exporta launch_sitl.sh) y la raíz del workspace deducida del
paquete instalado: así `model: models/yolo11n.engine` funciona en la Jetson
aunque se lance desde otra carpeta.

Python puro, testeable con pytest.
"""
import os


def resolve_model_path(path, roots):
    """Primera ruta existente para `path`; None si no está en ningún lado."""
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    for root in roots:
        for cand in (os.path.join(root, path), os.path.join(root, 'models', path)):
            if os.path.exists(cand):
                return cand
    return None


def is_downloadable(path):
    """Nombre oficial de ultralytics (ej. yolo11n.pt): si no está, lo descarga."""
    return path.endswith('.pt') and not os.path.dirname(path)
