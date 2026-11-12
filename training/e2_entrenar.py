#!/usr/bin/env python3
"""Etapa 2: entrenamiento de YOLO sobre los tiles de VisDrone.

Toda la configuración común vive en `entrenar()`. extra_tuning.py la importa para las
pruebas cortas y este mismo archivo la usa desde la terminal para el
entrenamiento final. Así el tuning y el entrenamiento final comparten imgsz,
semilla, batch, etc., y lo que se aprende en el tuning vale para el final.

Al terminar escribe `ficha_modelo.json` junto a los pesos: qué datos, qué
hiperparámetros, qué métricas en val y con qué versiones se entrenó.

Los pesos base (yolo11n.pt, yolo11s.pt) se buscan en pesos_base/; si no están,
Ultralytics los baja ahí mismo.

Uso (los dos modelos actuales, n_final y s_final):
  python e2_entrenar.py --data ~/datasets/visdrone_tiles/data.yaml --model yolo11n.pt --name n_final
  python e2_entrenar.py --data ... --model yolo11s.pt --name s_final
  python e2_entrenar.py --data ... --name n_final --resume      # retomar si se cortó
  python e2_entrenar.py --data ... --model yolo11n.pt --name n_tuning --cambios runs/tuning/ganador.json
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

PESOS_BASE = Path(__file__).resolve().parent / 'pesos_base'


def resolver_pesos(pesos: str) -> str:
    """`yolo11n.pt` -> `pesos_base/yolo11n.pt`. Una ruta (o un archivo que ya existe
    tal cual) se deja igual. Si el de pesos_base/ falta, Ultralytics lo baja a esa
    misma ruta."""
    if '/' in pesos or Path(pesos).exists():
        return pesos
    PESOS_BASE.mkdir(exist_ok=True)
    return str(PESOS_BASE / pesos)


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
        model = YOLO(resolver_pesos(modelo))
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
        'metricas_test': None,          # las completa e3_evaluar.py (etapa 3)
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
    ap = argparse.ArgumentParser(description='Etapa 2: entrenamiento.')
    ap.add_argument('--data', required=True, type=Path, help='data.yaml de e1_preparar_datos.py')
    ap.add_argument('--model', default='yolo11n.pt', help='yolo11n.pt | yolo11s.pt')
    ap.add_argument('--name', required=True, help='nombre del run, p. ej. n_final')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=15,
                    help='cortar si el mAP de val no mejora en N épocas')
    ap.add_argument('--cambios', type=Path,
                    help='JSON con los hiperparámetros ganadores (ganador.json de extra_tuning.py)')
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
