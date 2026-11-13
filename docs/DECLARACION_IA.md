# Declaración de uso de IA generativa

> Las secciones de verificación, decisiones y cambios humanos las escribió el equipo.
> Lo demás sale del historial de git y se puede verificar.

## Equipo

Integrantes y aporte principal de cada uno:

| Integrante | Aporte principal |
|---|---|
| Ignacio Ibañez | Programador: código del sistema y pruebas en la simulación de vuelo (SITL) |
| Gerardo González | Entrenamiento del modelo que detecta personas: datos, entrenamiento y resultados |
| Cristofer Jeria | Mapa y alertas: cómo funciona el sistema completo (arquitectura) |

## Herramienta usada

- **Claude Code** (Anthropic), modelo Claude Opus 5.5, como asistente de programación
  en la terminal y en VS Code.

## Alcance del uso (verificable en git)

**21 de los 25 commits del repo** (`git log --no-merges`, del 2026-10-28 al 2026-11-12)
tienen el trailer `Co-Authored-By: Claude Opus 5.5`; los otros 4 son ediciones de
documentación hechas a mano en GitHub. Es decir: el uso fue **material** y abarca todo el
código. Para verlo:

```bash
git log --format='%h %ad %an | %(trailers:key=Co-Authored-By,valueonly)' --date=short
```

En qué se usó:

| Área | Uso de la IA |
|---|---|
| Código ROS 2 y lógica (`src/person_yolo/`) | Escritura de nodos, tiling, tracker, geolocalización, mapa web, a partir de lo que el equipo pidió |
| Tests (`src/person_yolo/test/`) | Escritura de los tests unitarios |
| Entrenamiento (`training/`) | Scripts de preparación de datos, entrenamiento y evaluación. **El entrenamiento y la evaluación los ejecutó el equipo en su GPU**; las métricas salen de esas corridas |
| Revisión de código | Revisión de la rama `entrenamiento-visdrone` (2026-11-10). Encontró, entre otros, el problema de la web descrito en [RIESGOS_Y_ETICA.md](RIESGOS_Y_ETICA.md) §4 |
| Documentación | Redacción de README y `docs/` (2026-11-10), a partir de los resultados reales |
| Correcciones de la revisión (2026-11-12) | Seguridad de la web, personas contadas dos veces, fix tardío que recreaba una persona fusionada, columna `id` de `personas.csv`, finales de línea (`.gitattributes`), con sus tests |
| `entrega_eval01.zip` (material externo, 2026-11-12) | Paquete de documentación para EVAL-01 generado con IA **fuera de este repo**, sobre el `main` viejo. Se revisó y **no se aplicó**: duplicaba herramientas existentes y afirmaba cosas ya falsas (p. ej. "no se entrena"). Se rescataron, adaptadas y verificadas contra el código: el guion de demo, el plan B y las preguntas de [DEFENSA.md](DEFENSA.md), la métrica "dentro de 2σ" de `verdad_sitl.py` y el aviso de licencia y correo personal |

## Qué no hizo la IA

- **No generó resultados.** Todas las métricas vienen de ejecuciones reales y su archivo
  de origen está citado:
  - `training/runs/` y `docs/resultados/` (VisDrone).
  - `~/yolo_flights/` y `docs/resultados/sitl/` (vuelos en SITL).
  - `pytest`.
- **No voló el dron.** El sistema no puede comandarlo (ningún nodo publica en
  `/fmu/in/*`): los vuelos en SITL se hicieron a mano.

## Cómo se verificó lo que produjo la IA

Lo verificable desde el repo:
- 153 tests unitarios en verde (`docs/resultados/pytest.txt`).
- Evaluación en VisDrone test-dev con el mismo `detect_persons()` que corre en vuelo
  (sin una copia aparte).
- Vuelos en SITL comparados con la posición real de las personas (`verdad_sitl.py`). Esa
  comparación **mostró problemas** (error de ubicación ~12 m, subestimado): se documentan
  en [VALIDACION.md](VALIDACION.md), no se ocultan.
- Las licencias de terceros se revisaron en sus repositorios oficiales
  ([RIESGOS_Y_ETICA.md](RIESGOS_Y_ETICA.md) §6).

Lo que revisó o probó el equipo:

- Volamos el SITL en cada prueba, revisando que el sistema detectara a las personas.
- Revisamos el recall de las cajas (*bounding boxes*) de cada modelo.
- Comparamos los puntos marcados en el mapa con la posición real de las personas en el
  mapa de Unreal, usando los archivos de verificación (`verdad.csv`, `verdad.geojson`,
  `informe.txt` en `docs/resultados/sitl/`).

## Decisiones y cambios humanos

Decisiones que tomó el equipo y no la IA:
- El problema a resolver y el alcance (vuelo manual, solo mirar, nunca comandar al dron).
- Usar VisDrone y reentrenar en lugar de quedarse con COCO.
- Entrenar dos modelos (`n` y `s`) y compararlos en lugar de uno solo: más esfuerzo,
  pero para elegir con evidencia y obtener mejores resultados.
- Revisar los hiperparámetros del entrenamiento (`training/e2_entrenar.py`: tamaño de
  imagen 640 igual al de los tiles y la Jetson, batch 8 por la memoria de la GPU, 50
  épocas, misma semilla en todos los runs). Las pruebas automáticas de tuning, cambiando
  un hiperparámetro a la vez (`training/extra_tuning.py`), quedaron preparadas pero no
  se ejecutaron por el tiempo que lleva cada entrenamiento.
- Elegir `n_final` para la Jetson.
- El hardware (Jetson Orin NX, lidar), la escena de Unreal y la colocación de las personas.

Propuestas que el equipo rechazó o cambió:

- **Radar para detectar la cercanía de las personas.** Se descartó: el radar funciona
  bien con blancos en el aire (aviones, drones, aves), pero es ineficiente para
  distinguir personas en tierra.
- **`entrega_eval01.zip`** (ver arriba): revisado y no aplicado.

## Límites

- La IA puede equivocarse con seguridad. Por eso cada cifra de la documentación apunta
  a un archivo generado por una ejecución, y lo no validado se marca como tal.
- Para entender lo que usa, el equipo revisó y preguntó cómo funciona cada nodo y cómo
  se relaciona con sus tópicos de ROS 2. Qué parte explica cada integrante:
  [DEFENSA.md](DEFENSA.md) §4.
