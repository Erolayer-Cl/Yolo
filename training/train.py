#!/usr/bin/env python3
"""Entrenamiento de YOLO sobre los tiles de VisDrone (etapas 2 y 3).

Toda la configuración común vive en `entrenar()`. tune.py la importa para las
pruebas cortas y este mismo archivo la usa desde la terminal para el
entrenamiento final. Así el tuning y el entrenamiento final comparten imgsz,
semilla, batch, etc., y lo que se aprende en el tuning vale para el final.

Al terminar escribe `ficha_modelo.json` junto a los pesos: qué datos, qué
hiperparámetros, qué métricas en val y con qué versiones se entrenó.

Uso (etapa 3):
  python train.py --data ~/datasets/visdrone_tiles/data.yaml --model yolo11n.pt --name n_sin_tuning
  python train.py --data ... --model yolo11n.pt --name n_con_tuning --cambios runs/tuning/ganador.json
  python train.py --data ... --model yolo11s.pt --name s_con_tuning --cambios runs/tuning/ganador.json
  python train.py --data ... --name n_sin_tuning --resume      # retomar si se cortó
"""
import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

# Configuración común a TODOS los entrenamientos. Cambiarla acá la cambia en el
# tuning y en el entrenamiento final a la vez.
COMUN = {
    'imgsz': 640,          # = tiles y = export de la Jetson (jetson.yaml)
    'seed': 0,             # misma semilla: las diferencias entre pruebas no son azar
    'deterministic': True,
    'batch': 8,            # RTX 4060 Ti de 8 GB: entra yolo11s; igual en TODOS los runs
    'workers': 8,
    'plots': True,         # curvas, matriz de confusión, ejemplos de val
    'exist_ok': False,     # no pisar un entrenamiento anterior por error
}

METRICAS = {               # nombre en Ultralytics -> nombre en la ficha
    'metrics/precision(B)': 'precision',
    'metrics/recall(B)': 'recall',
    'metrics/mAP50(B)': 'mAP50',
    'metrics/mAP50-95(B)': 'mAP50-95',
}

LIMITACIONES = [
    'Entrenado con fotos reales (VisDrone, mayormente urbano); el sistema por ahora '
    'corre en SITL con Unreal: puede haber salto de dominio.',
    'Entrenado con tiles de 640 a resolución nativa: pensado para la pasada de tiles '
    'del detector (inference.py), no para imágenes completas achicadas.',
    'El score no es una probabilidad calibrada.',
]


def entrenar(modelo, data, nombre, epocas, cambios=None, carpeta='runs/train',
             resume=False, **extra):
    """Entrena `modelo` y devuelve (métricas en val, carpeta del run).

    `cambios`: hiperparámetros que difieren de los por defecto, p. ej.
    {'scale': 0.25}. Es lo ÚNICO que debería variar entre pruebas.
    `extra`: opciones que no son hiperparámetros (p. ej. fraction para pruebas).
    """
    from ultralytics import YOLO

    cambios = dict(cambios or {})
    carpeta = Path(carpeta).resolve()
    if resume:
        last = carpeta / nombre / 'weights' / 'last.pt'
        if not last.exists():
            raise SystemExit(f'No hay {last} para retomar.')
        model = YOLO(str(last))
        resultados = model.train(resume=True)
    else:
        args = {**COMUN, **extra, **cambios, 'data': str(data), 'epochs': epocas,
                'project': str(carpeta), 'name': nombre}
        model = YOLO(modelo)
        resultados = model.train(**args)
    # La carpeta real del run: si `nombre` ya existía, Ultralytics usa nombre2, nombre3...
    run_dir = Path(model.trainer.save_dir)

    # Ultralytics termina validando best.pt sobre val: esas son las métricas.
    # Ojo: precisión y recall están al umbral de confianza que maximiza F1.
    metricas = {nuevo: round(float(resultados.results_dict[viejo]), 4)
                for viejo, nuevo in METRICAS.items()}
    escribir_ficha(run_dir, modelo, data, epocas, cambios, extra, metricas)
    return metricas, run_dir


def escribir_ficha(run_dir, modelo, data, epocas, cambios, extra, metricas):
    """Ficha del modelo (lámina 7 de la clase): el .pt solo no alcanza."""
    import torch
    import ultralytics

    stats = Path(data).parent / 'stats.json'
    ficha = {
        'fecha': datetime.now().isoformat(timespec='seconds'),
        'pesos': str(run_dir / 'weights' / 'best.pt'),
        'modelo_base': str(modelo),
        'clases': {0: 'person'},
        'dataset': {
            'data_yaml': str(Path(data).resolve()),
            'origen': 'VisDrone2019-DET, pedestrian + people -> person, tiles de 640',
            'preparacion': json.loads(stats.read_text()) if stats.exists() else None,
        },
        'entrenamiento': {'epocas': epocas, 'comun': COMUN, 'cambios': cambios,
                          'extra': extra},
        'metricas_val': metricas,
        'metricas_test': None,          # las completa evaluate.py (etapa 4)
        'entorno': {
            'python': platform.python_version(),
            'ultralytics': ultralytics.__version__,
            'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        },
        'limitaciones': LIMITACIONES,
    }
    (run_dir / 'ficha_modelo.json').write_text(json.dumps(ficha, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description='Entrenamiento final (etapa 3).')
    ap.add_argument('--data', required=True, type=Path, help='data.yaml de prepare_visdrone.py')
    ap.add_argument('--model', default='yolo11n.pt', help='yolo11n.pt | yolo11s.pt')
    ap.add_argument('--name', required=True, help='nombre del run, p. ej. n_sin_tuning')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=15,
                    help='cortar si el mAP de val no mejora en N épocas')
    ap.add_argument('--cambios', type=Path,
                    help='JSON con los hiperparámetros ganadores (ganador.json de tune.py)')
    ap.add_argument('--batch', type=int, default=COMUN['batch'])
    ap.add_argument('--project', default='runs/train')
    ap.add_argument('--resume', action='store_true', help='retomar desde last.pt')
    a = ap.parse_args()

    cambios = json.loads(a.cambios.read_text())['cambios'] if a.cambios else {}
    COMUN['batch'] = a.batch
    metricas, run_dir = entrenar(a.model, a.data.expanduser(), a.name, a.epochs, cambios,
                                 a.project, a.resume, patience=a.patience)
    print(f'\nMétricas en val: {metricas}\nFicha: {run_dir / "ficha_modelo.json"}')


if __name__ == '__main__':
    main()
