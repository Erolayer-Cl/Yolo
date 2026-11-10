# Entrenamiento del detector de personas lejanas (VisDrone)

Scripts para reentrenar el YOLO del repo `yolo` (paquete `person_yolo`) con
personas vistas desde dron, y medir si mejora frente al modelo COCO actual.

## Requisitos

- Python 3.10+ y las dependencias de [`requirements.txt`](requirements.txt): versiones
  exactas del venv con el que se entrenó (`~/venvs/yolo-train`, Python 3.12.3).
  ```bash
  python3 -m venv ~/venvs/yolo-train && . ~/venvs/yolo-train/bin/activate
  pip install -r requirements.txt     # torch con CUDA: ver el comentario del archivo
  ```
- Estos scripts importan `make_tiles()` y `detect_persons()` del paquete `person_yolo`
  de este mismo repo, para usar exactamente el mismo criterio que el detector en vuelo.
  No hace falta ROS ni compilar: son módulos de Python puro.
- Los datasets van FUERA del repo (por ejemplo `~/datasets/`); `runs/` y los `.pt`
  quedan ignorados por git. Los resultados que importan están copiados en
  [`docs/resultados/`](../docs/resultados/).
- GPU: se usó una RTX 4060 de 8 GB (`batch=8` en todos los entrenamientos).

## Datos y licencias

| | |
|---|---|
| Dataset | **VisDrone2019-DET** — fotos y videos tomados desde drones en ciudades de China, con cajas anotadas a mano. <https://github.com/VisDrone/VisDrone-Dataset> |
| Divisiones usadas | `train` (6471 imágenes), `val` (548), `test-dev` (1610). `test-challenge` no se usa: no publica sus etiquetas |
| Clases | `pedestrian` (persona de pie o caminando) + `people` (persona sentada, agachada, en grupo) → una sola clase `person`. El resto (autos, bicicletas…) se descarta |
| Condición de uso | El README oficial **no declara una licencia**. Solo dice que *"Researchers can use test-dev to publish papers"*. Se usa con fines académicos, sin redistribuir el dataset (va fuera del repo). Las 3 imágenes de ejemplo en `docs/resultados/img/` son de test-dev |
| Cita | P. Zhu, L. Wen, D. Du, X. Bian, H. Fan, Q. Hu, H. Ling, "Detection and Tracking Meet Drones Challenge", *IEEE TPAMI*, vol. 44, no. 11, pp. 7380–7399, 2021 |
| Pesos base | `yolo11n.pt` y `yolo11s.pt` de Ultralytics, preentrenados en COCO. Ultralytics se distribuye bajo **AGPL-3.0** (uso académico permitido; uso comercial requiere su licencia Enterprise) |

Lo que hace `prepare_visdrone.py` (corrida real, de la `ficha_modelo.json`; parámetros
`tile 640, overlap 0.15, min_px 4, bg_ratio 0.1, seed 0`):

| División | Imágenes | Tiles guardados | con personas | vacíos | Personas | <16 px | 16–32 px | ≥32 px |
|---|---|---|---|---|---|---|---|---|
| train | 6471 | 29 653 (de 44 205) | 26 688 | 2965 | 217 550 | 139 345 | 62 460 | 15 745 |
| val | 548 | 2842 | 2535 | 307 | 33 592 | 23 660 | 8893 | 1039 |
| test | 1610 | 9838 | 5690 | 4148 | 59 261 | 49 376 | 8134 | 1751 |

El **64 % de las personas de train mide menos de 16 px** (lado menor de la caja): es
justo el caso que COCO resuelve mal y por el que se reentrena.

## Orden de ejecución

Desde esta carpeta (`training/`), con el repo como `--repo`. Son los comandos que se
corrieron para los pesos actuales (`n_final`, `s_final`, 2026-11-09):

```bash
cd training
REPO=..
DATA=~/datasets/visdrone_tiles/data.yaml

# 1. Datos: baja VisDrone (~2 GB) y lo corta en tiles de 640 con una sola clase, person
python prepare_visdrone.py --repo $REPO --src ~/datasets/VisDrone --out ~/datasets/visdrone_tiles --download

# 2. Entrenamiento: 50 épocas, patience 15, sin cambios de hiperparámetros
python train.py --data $DATA --model yolo11n.pt --name n_final
python train.py --data $DATA --model yolo11s.pt --name s_final
#    -> runs/train/<nombre>/weights/best.pt + ficha_modelo.json (datos, entorno y métricas)

# 3. Evaluación en test-dev: COCO actual vs. los dos modelos nuevos
python evaluate.py --repo $REPO --data $DATA --src ~/datasets/VisDrone \
    --model coco=yolo11n.pt \
    --model n_final=runs/train/n_final/weights/best.pt \
    --model s_final=runs/train/s_final/weights/best.pt
#    -> runs/eval/resumen.md: nivel 1 (tiles) y nivel 2 (sistema) con sitl.yaml y con jetson.yaml
#       val_<modelo>/ (curvas) y ejemplos_<config>_<modelo>/ (imágenes)

# 4. Usarlos en el detector
cp runs/train/n_final/weights/best.pt ../models/visdrone_n.pt
cp runs/train/s_final/weights/best.pt ../models/visdrone_s.pt
```

**No ejecutado (previsto):** `tune.py` prueba hiperparámetros de a uno (A por defecto,
B `scale 0.25`, C `flipud 0.5`, D = B+C) y escribe `runs/tuning/ganador.json`, que se pasa a
`train.py --cambios`. No se corrió por tiempo: los dos modelos actuales usan los valores por
defecto de Ultralytics (`"cambios": {}` en su ficha).

```bash
python tune.py --data $DATA          # -> runs/tuning/tabla.md y ganador.json
python train.py --data $DATA --model yolo11n.pt --name n_tuning --cambios runs/tuning/ganador.json
```

## Resultados (VisDrone test-dev, 1610 imágenes)

Copia en [`docs/resultados/resumen.md`](../docs/resultados/resumen.md).

- **Nivel 1:** el modelo solo, sobre tiles (Ultralytics val).
- **Nivel 2:** el sistema completo, con `detect_persons()` y los umbrales reales de
  `sitl.yaml` (frame completo + tiles + fusión), sobre imágenes enteras.

| Modelo | mAP50 tiles | Precisión | Recall | Recall <16 px | Recall 16–32 px | Recall ≥32 px | F1 | Falsas alarmas/img | ms/img (RTX 4060) |
|---|---|---|---|---|---|---|---|---|---|
| COCO `yolo11n` | 0.184 | 0.775 | 0.119 | 0.050 | 0.335 | 0.669 | 0.207 | 0.59 | 24 |
| `n_final` | 0.463 | 0.685 | 0.448 | 0.373 | 0.741 | 0.791 | 0.542 | 3.51 | 25 |
| `s_final` | 0.508 | 0.650 | 0.508 | 0.437 | 0.782 | 0.845 | 0.570 | 4.66 | 36 |

Con `jetson.yaml` (overlap 0.15) los números cambian en el tercer decimal.

**Qué dicen los números:**
- El recall total sube de 0.12 a 0.45 (`n`) y 0.51 (`s`): unas **4 veces más
  personas encontradas**. La mejora está en las chicas: de 5 % a 37–44 % en las de
  menos de 16 px.
- A cambio, la **precisión baja** (0.78 → 0.65–0.69) y las **falsas alarmas pasan de 0.6
  a 3.5–4.7 por imagen**. En vuelo no todas llegan al mapa: el tracker pide `min_hits`
  frames y `min_seen_sec` de seguimiento antes de crear un waypoint.
- `s` gana ~6 pts de recall frente a `n`, pero tarda ~45 % más (36 vs 25 ms en la
  4060) y da más falsas alarmas.

**Decisión:** `n_final` para el dron real (Jetson Orin NX: el tiempo por frame limita
cuántos tiles por segundo se procesan) y como peso por defecto para probar en SITL.
`s_final` queda como alternativa en PC cuando sobra GPU. En la Jetson no se midió
todavía: las cifras de ms son de la RTX 4060.

**Interpretación con cuidado:**
- Es un test con fotos reales de VisDrone, no del SITL ni del dron. En Unreal no hay
  etiquetas: la comparación ahí es cualitativa (`comparar.py`) o por posición
  (`scripts/verdad_sitl.py`, ver [docs/VALIDACION.md](../docs/VALIDACION.md)).
- Parte de las "falsas alarmas" pueden ser personas que VisDrone no anotó. No se revisó
  a mano.

```bash
# 5. Comparar en el SITL (sin etiquetas): los mismos frames por cada modelo
ros2 bag record /camera -o ~/bags/vuelo1        # mientras se vuela en Unreal
python comparar.py --src ~/bags/vuelo1           # coco, ../models/visdrone_n.pt y visdrone_s.pt
#    -> runs/comparar/vuelo1/: comparacion.mp4 (mosaico), resumen.md, discrepancias/
#       también acepta un video o una carpeta de imágenes; --model NOMBRE=PESOS para elegir
```

Si un paso se corta: `tune.py` salta las pruebas ya terminadas al relanzarlo, y
`train.py --name <nombre> --resume` retoma un entrenamiento desde `last.pt`.

## Archivos

| Archivo | Etapa | Qué hace |
|---|---|---|
| `prepare_visdrone.py` | 1 | VisDrone -> tiles de 640 en formato YOLO, clase única `person` |
| `train.py` | 2 | Configuración común (`COMUN`) y `entrenar()`; escribe `ficha_modelo.json` |
| `tune.py` | (no ejecutado) | Pruebas de hiperparámetros, una a la vez, medidas en val |
| `evaluate.py` | 3 | Métricas por tiles y del sistema completo, ejemplos visuales |
| `comparar.py` | 5 | Modelos lado a lado sobre frames del SITL (rosbag, video o imágenes), sin etiquetas |

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
- Sin tuning de hiperparámetros: los valores son los de Ultralytics por defecto.
- Una sola semilla (0) por modelo: no hay barras de error entre corridas.
