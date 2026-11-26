#!/usr/bin/env python3
"""Repite el nivel 2 de e3_evaluar.py (sistema completo con sitl.yaml) para un peso, en CPU.

Sirve para verificar las cifras de VisDrone test-dev sin GPU ni los tiles de la etapa 1:
solo necesita las imágenes y anotaciones de VisDrone2019-DET-test-dev. Usa la misma
eval_system() de e3_evaluar.py y detect_persons() del nodo.

Uso:
  python extra_reevaluar_cpu.py --src ~/datasets/VisDrone --nombre n_final \\
      --pesos ../models/visdrone_n.pt --out ../docs/resultados/reevaluacion_cpu
"""
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'training'))
import e3_evaluar as e3  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--src', required=True, type=Path, help='carpeta con VisDrone2019-DET-test-dev')
    ap.add_argument('--nombre', required=True)
    ap.add_argument('--pesos', required=True)
    ap.add_argument('--limit', type=int, default=0, help='solo N imágenes (pruebas)')
    ap.add_argument('--out', type=Path, default=Path('runs/reevaluacion_cpu'))
    a = ap.parse_args()
    from ultralytics import YOLO
    fns = e3.import_inference(REPO)
    params, iou_nms, names = e3.load_params(REPO / 'src/person_yolo/config/sitl.yaml', fns[0])
    imgs = sorted((a.src.expanduser() / 'VisDrone2019-DET-test-dev' / 'images').glob('*.jpg'))
    if a.limit:
        imgs = imgs[:a.limit]
    ejemplos = set(random.Random(0).sample(imgs, min(6, len(imgs))))
    args = SimpleNamespace(device='cpu', iou=0.5, out=a.out)
    model = YOLO(a.pesos)
    r = e3.eval_system(model, a.nombre, args, imgs, ejemplos, params, iou_nms,
                       e3.person_classes(model, names), fns, 'sitl')
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f'{a.nombre}.json').write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
