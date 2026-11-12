# Declaración de uso de IA generativa

> **Para completar por el equipo.** Las secciones marcadas con ✏️ las tiene que escribir
> el equipo: solo ustedes saben qué decidieron, qué revisaron y qué cambiaron. Lo
> demás sale del historial de git y se puede verificar.

## Equipo

✏️ Integrantes y aporte principal de cada uno:

| Integrante | Usuario de git | Aporte principal |
|---|---|---|
| ✏️ | `Erolayer_CL` | ✏️ |
| ✏️ Ignacio Ibañez | `Ignacio Ibañez` | ✏️ |
| ✏️ Gerardo González| gerardogonzalez0264| ✏️Análisis de datos|

## Herramienta usada

- **Claude Code** (Anthropic), modelo Claude Opus 5.5, como asistente de programación
  en la terminal y en VS Code.

## Alcance del uso (verificable en git)

**Los 13 commits del repo** (`git log`, del 2026-10-28 al 2026-11-10) tienen el trailer
`Co-Authored-By: Claude Opus 5.5`. Es decir: el uso fue **material** y abarca todo el
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
- 140 tests unitarios en verde (`docs/resultados/pytest.txt`).
- Evaluación en VisDrone test-dev con el mismo `detect_persons()` que corre en vuelo
  (sin una copia aparte).
- Vuelos en SITL comparados con la posición real de las personas (`verdad_sitl.py`). Esa
  comparación **mostró problemas** (error de ubicación ~12 m, subestimado): se documentan
  en [VALIDACION.md](VALIDACION.md), no se ocultan.
- Las licencias de terceros se revisaron en sus repositorios oficiales
  ([RIESGOS_Y_ETICA.md](RIESGOS_Y_ETICA.md) §6).

✏️ Lo que revisó o probó el equipo (ejemplos: "volamos el SITL después de cada cambio",
"revisamos los números del informe contra resumen.md", "probamos la web desde el
celular"):

- ✏️
- ✏️

## Decisiones y cambios humanos

✏️ Decisiones que tomó el equipo y no la IA. Ejemplos a confirmar o corregir:
- El problema a resolver y el alcance (vuelo manual, solo mirar, nunca comandar al dron).
- Usar VisDrone y reentrenar en lugar de quedarse con COCO.
- Elegir `n_final` para la Jetson.
- El hardware (Jetson Orin NX, lidar), la escena de Unreal y la colocación de las personas.

✏️ Propuestas de la IA que el equipo rechazó o cambió:

- ✏️

## Límites

- La IA puede equivocarse con seguridad. Por eso cada cifra de la documentación apunta
  a un archivo generado por una ejecución, y lo no validado se marca como tal.
- ✏️ Cómo aseguró el equipo entender lo que usa (para la defensa): qué partes puede
  explicar cada integrante.
