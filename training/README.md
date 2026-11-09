# Entrenamiento del detector de personas lejanas (VisDrone)

Scripts para reentrenar el YOLO del repo `yolo` (paquete `person_yolo`) con
personas vistas desde dron, y medir si mejora frente al modelo COCO actual.

## Requisitos

- Python 3.10+, `pip install ultralytics` (trae opencv, PyYAML, torch).
- Estos scripts importan `make_tiles()` y `detect_persons()` del paquete `person_yolo`
  de este mismo repo, para usar exactamente el mismo criterio que el detector en vuelo.
  No hace falta ROS ni compilar: son módulos de Python puro.
- Los datasets van FUERA del repo (por ejemplo `~/datasets/`); `runs/` y los `.pt`
  quedan ignorados por git.
- GPU: pensado para una RTX 4060 Ti de 8 GB (`batch=8` en todos los entrenamientos).

## Orden de ejecución

Desde esta carpeta (`training/`), con el repo como `--repo`:

```bash
cd training
REPO=..
DATA=~/datasets/visdrone_tiles/data.yaml

# 1. Datos: baja VisDrone (~2 GB) y lo corta en tiles de 640 con una sola clase, person
python prepare_visdrone.py --repo $REPO --src ~/datasets/VisDrone --out ~/datasets/visdrone_tiles --download

# 2. Tuning manual: pruebas A (por defecto), B (scale 0.25), C (flipud 0.5) y D (B+C, si ambas ayudan)
python tune.py --data $DATA
#    -> runs/tuning/tabla.md y runs/tuning/ganador.json

# 3. Entrenamiento final: n sin tuning, n con tuning y s con tuning
python train.py --data $DATA --model yolo11n.pt --name n_sin_tuning
python train.py --data $DATA --model yolo11n.pt --name n_con_tuning --cambios runs/tuning/ganador.json
python train.py --data $DATA --model yolo11s.pt --name s_con_tuning --cambios runs/tuning/ganador.json
#    -> runs/train/<nombre>/weights/best.pt + ficha_modelo.json

# 4. Evaluación en test-dev: COCO actual vs. los tres modelos nuevos
python evaluate.py --repo $REPO --data $DATA --src ~/datasets/VisDrone \
    --model coco=yolo11n.pt \
    --model n_sin_tuning=runs/train/n_sin_tuning/weights/best.pt \
    --model n_con_tuning=runs/train/n_con_tuning/weights/best.pt \
    --model s_con_tuning=runs/train/s_con_tuning/weights/best.pt
#    -> runs/eval/resumen.md: nivel 1 (tiles) y nivel 2 (sistema) con sitl.yaml y con jetson.yaml
#       val_<modelo>/ (curvas) y ejemplos_<config>_<modelo>/ (imágenes)
```

Si un paso se corta: `tune.py` salta las pruebas ya terminadas al relanzarlo, y
`train.py --name <nombre> --resume` retoma un entrenamiento desde `last.pt`.

## Archivos

| Archivo | Etapa | Qué hace |
|---|---|---|
| `prepare_visdrone.py` | 1 | VisDrone -> tiles de 640 en formato YOLO, clase única `person` |
| `train.py` | 2 y 3 | Configuración común (`COMUN`) y `entrenar()`; entrenamiento final desde la terminal |
| `tune.py` | 2 | Pruebas de hiperparámetros, una a la vez, medidas en val |
| `evaluate.py` | 4 | Métricas por tiles y del sistema completo, ejemplos visuales |

## Decisiones

- **Tiles de 640 a resolución nativa** con el `make_tiles()` del repo: en vuelo, las
  personas lejanas se buscan así. Entrenar con imágenes completas achicadas las
  dejaría 2-3 veces más chicas que en vuelo.
- **pedestrian + people -> `person`.** Regiones ignoradas de VisDrone tapadas con gris.
- **Personas cortadas por el borde del tile se etiquetan**; se descartan las astillas de
  menos de 4 px (solo en personas cortadas).
- **10 % de tiles vacíos en train**; en val y test se guardan todos (proporción real).
- **Divisiones oficiales:** train para entrenar, val para tuning y para cortar el
  entrenamiento, test-dev solo para la evaluación final.
- **Mismos `imgsz=640`, semilla y batch en todos los entrenamientos** (`COMUN` en `train.py`).
- **La evaluación del sistema usa `detect_persons()` y los parámetros de `sitl.yaml` y de
  `jetson.yaml`** (lo que corre hoy y el dron real): mide lo que vuela, no una copia.

## Limitaciones

- VisDrone son fotos reales y mayormente urbanas; el sistema por ahora corre en SITL con
  Unreal. Puede haber salto de dominio.
- En vuelo los tiles corren cada `every_n_frames`; la evaluación los usa en todas las
  imágenes (mejor caso por imagen).
- Las métricas por tiles cuentan a una persona varias veces si aparece en tiles solapados;
  por eso importa el nivel 2 (imágenes completas).
- El score de YOLO no es una probabilidad calibrada.
- Los ms por imagen se miden en la GPU donde se evalúa, no en la Jetson.
