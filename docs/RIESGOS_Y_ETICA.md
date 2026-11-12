# Riesgos, ética, seguridad y licencias

Riesgos de **este** sistema (detector de personas en un dron, para búsqueda y rescate)
y qué hace o no hace para mitigarlos. Donde la mitigación todavía no está, se dice.

## 1. Riesgo para las personas buscadas

| Riesgo | Por qué pasa | Mitigación | Estado |
|---|---|---|---|
| **No detectar a alguien** (falso negativo) y que se dé una zona por revisada | Recall de 0.45 en VisDrone con `n_final`: más de la mitad de las personas reales no se detecta, sobre todo las de < 16 px (recall 0.37). Ver [VALIDACION.md](VALIDACION.md) | El sistema **asiste** al observador, no lo reemplaza. El piloto sigue viendo el video FPV completo. El botón **Marcar aquí** agrega a quien el detector no marcó | Hay que decirlo explícitamente en la operación: "sin alerta" **no** significa "no hay nadie" |
| **Mandar al equipo a un lugar equivocado** | En SITL, el error real de ubicación fue de ~12 m (mediana), mientras el sistema informaba ±1–3 m. La persona real cayó dentro de la zona de 2σ en 0 de 4 puntos ubicados del vuelo 4 (calibrado serían ~86 %; `resultados/sitl/20261111_160740/informe.txt`) | Cada punto lleva su error y una zona de búsqueda de 2σ. Las aproximadas se marcan aparte y nunca dan coordenadas de rescate (`lat/lon = null`) | ⚠️ El error informado está **subestimado**. Pendiente calibrar (`calibrar_geo.py`) antes de confiar en el radio |
| **Falsas alarmas** que gastan tiempo del equipo | 3.5 falsas alarmas por imagen con `n_final` (0.6 con COCO). En SITL, 2 personas inexistentes con el dron en tierra | Tracker con `min_hits` y `min_seen_sec: 2 s` antes de crear un waypoint; score visible en la tabla | Parcial. Se puede subir `tiling.conf` (hoy 0.25) y medir el efecto |
| **Contar dos veces a la misma persona** | Sale de cuadro y vuelve; aproximadas que se repiten | Deduplicación por distancia y fusión de marcas (`persons.py`) | 2–3 duplicados por vuelo en SITL, también con la fusión de la versión actual (vuelo 4: 1). Las personas de la escena están a <15 m entre sí, lo que lo hace difícil de medir |

## 2. Sesgo de los datos

- **VisDrone es urbano y de China**: calles, plazas, ropa urbana, luz de ciudad. Un caso de
  rescate típico (montaña, bosque, nieve, agua, persona tendida o tapada) **no está
  representado**. El recall en esos casos puede ser mucho menor y no está medido.
- **Pose:** las clases son `pedestrian` (de pie o caminando) y `people` (sentado, agachado).
  Una persona **acostada** —el caso más importante en rescate— casi no aparece en el
  entrenamiento.
- **Dominio simulado:** la validación de sistema es en Unreal, con 3 actores quietos. No
  dice nada sobre personas reales, en movimiento o parcialmente tapadas.
- **No se midió el desempeño por grupo** (ropa, color de piel, edad). Con fotos aéreas de
  pocos píxeles esas variables casi no se distinguen, pero el contraste ropa/fondo sí
  importa, y no está evaluado.

## 3. Privacidad

El sistema graba y transmite imágenes de personas y su ubicación.

| Dato | Dónde queda | Medida |
|---|---|---|
| Video de la cámara | Solo en memoria. No se graba, salvo que alguien grabe un rosbag a mano | En la Jetson, `foxglove_bridge` tiene `topic_whitelist`: la imagen cruda nunca sale por WiFi. La anotada sale a media resolución |
| Video FPV (`:8080`) e imagen anotada | Se transmiten por HTTP **sin cifrar** en la red del dron | Usar una red WiFi propia con contraseña, no una abierta |
| Posición de cada persona detectada | `~/yolo_flights/<fecha>/personas.csv`, `trail.kml`, `/rescue/waypoints` | Es dato personal (ubicación de una persona identificable en un lugar y momento). Borrar los vuelos que no sean necesarios. No subirlos a repositorios públicos |
| Rosbags de prueba | Donde los guarde quien prueba (`~/bags/`) | No se versionan. En este repo solo hay vuelos **simulados** (`docs/resultados/sitl/`), sin personas reales |

En un vuelo real sobre terceros que no son el objetivo de la búsqueda, aplica la
regulación local de drones y de datos personales. En Chile son la norma DAN 151 de la DGAC
y la ley de protección de datos personales (Ley 19.628, en proceso de reemplazo por la
Ley 21.719). Este repo no evalúa el cumplimiento; hay que revisarlo antes de volar.

## 4. Seguridad

| Riesgo | Medida | Estado |
|---|---|---|
| Que el sistema **comande al dron** por error o por un ataque | Regla de diseño: **ningún nodo publica en `/fmu/in/*`**. Solo se lee `/fmu/out/*` | ✅ Se cumple en todo el código |
| Que alguien en el WiFi publique en ROS desde Foxglove | `jetson.yaml`: `foxglove_bridge` en solo lectura (`capabilities: [connectionGraph]`) | ✅ |
| Web de rescate (`:8081`) **sin contraseña**: cualquiera en la red puede ver las coordenadas y agregar o borrar marcas manuales | `web_marks: false` apaga los botones; `web_port: 0` apaga la web | ⚠️ Sin contraseña por diseño, para operar sin configuración: hay que usar una red WiFi propia. Lo que encontró la revisión de código está **corregido** (2026-11-12, `web_map.py`, con tests en `test_web_map.py`): los POST exigen `Content-Type: application/json` y, si traen `Origin`, que sea el mismo host, así una página de **otro sitio** abierta en el mismo navegador no puede mandar marcas (CSRF); y un `Content-Length` negativo devuelve 400 en vez de colgar un hilo |
| Secretos en el repo | No hay credenciales, tokens ni claves. `settings.json` de AirSim no tiene ninguno | ✅ Revisado |

## 5. Impacto

- **Positivo:** cubre más área que un observador mirando el video, y deja coordenadas que
  se pueden pasar al equipo en tierra.
- **Riesgo de confianza excesiva:** un mapa con puntos rojos y radios en metros parece
  más preciso de lo que es (ver §1). La interfaz muestra el error, pero hoy ese error
  está subestimado.
- **Uso indebido:** el mismo sistema sirve para vigilar personas. El repo no lo impide.
  El diseño de solo mirar, sin grabar video, reduce lo que queda guardado.

## 6. Licencias de terceros

Revisadas en la fuente el 2026-11-10.

| Componente | Licencia | Implicancia para este proyecto |
|---|---|---|
| Ultralytics (código y pesos `yolo11n/s.pt`) | **AGPL-3.0**; alternativa comercial: licencia Enterprise. <https://github.com/ultralytics/ultralytics> | Uso académico permitido. Si se distribuye o se ofrece como servicio, el código propio que lo usa tiene que publicarse bajo AGPL-3.0, o hay que comprar la licencia Enterprise |
| VisDrone2019-DET | **Sin licencia declarada** en su README. Solo dice *"Researchers can use test-dev to publish papers"*. <https://github.com/VisDrone/VisDrone-Dataset> | Uso académico, citando el paper. El dataset no se redistribuye. Las 3 imágenes de ejemplo de test-dev en `docs/resultados/img/` se incluyen como evidencia de evaluación |
| PX4-Autopilot y `px4_msgs` (vendorizado en `src/px4_msgs/`) | BSD-3-Clause | Mantener el aviso de copyright en `src/px4_msgs/` |
| Cosys-AirSim | MIT (incluye los avisos de Microsoft AirSim, Codex Labs y Universidad de Amberes) | Solo se usa, no se redistribuye |
| ROS 2 (Jazzy/Humble), foxglove_bridge | Apache-2.0 / MIT (no verificado en la fuente) | Solo se usan |

El repo **no tiene licencia propia** (no hay `LICENSE`), y `package.xml` y `setup.py`
declaran **Apache-2.0**, que no es coherente con usar Ultralytics (AGPL-3.0). Si se
publica, la más simple y compatible es **AGPL-3.0** (agregar `LICENSE` y cambiar las dos
declaraciones). La otra opción es dejar Apache-2.0 para el código propio y aclarar en el
README que, al usarlo con Ultralytics, el conjunto queda bajo AGPL-3.0. Lo decide el equipo.
