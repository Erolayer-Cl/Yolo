# Demo y defensa

Guion de la demostración (EVAL-01, 13-11-2026), plan B si falla el entorno, y preguntas
probables de la defensa con el archivo de donde sale cada respuesta.

Regla de la rúbrica: toda cifra que se diga tiene que tener su archivo de origen. Las de
abajo lo tienen. Si algo no se midió, se dice que no se midió.

> Basado en parte en material de preparación generado con IA fuera de este repo
> (`entrega_eval01.zip`, `docs/09` y `06` §6.4–6.5), adaptado a esta rama y verificado
> contra el código. Ver [DECLARACION_IA.md](DECLARACION_IA.md).

## 1. Guion de la demo (entrada → proceso → resultado)

Antes: Unreal abierto con el proyecto `Pruebas` (mapa World_Demo), QGroundControl con la
radio y `make test` en verde.

1. **Arranque:** `make sitl MODEL=visdrone_n.pt`, Play en Unreal y Enter. Mostrar las
   ventanas de tmux: PX4, XRCE, airsim_node y la demo.
2. **Entrada:** despegar a mano y volar hacia las personas. Abrir el FPV en
   <http://localhost:8080>.
3. **Proceso:** se ven las cajas de YOLO, **verdes** = frame completo, **naranjas** =
   tile (personas lejanas), ver `overlay.py`. El conteo sube una vez por persona, no
   una vez por frame (`tracker.py` y `persons.py`).
4. **Resultado:** en el mapa de rescate (<http://localhost:8081>) aparece cada persona
   con su lat/lon y su zona de búsqueda. Las aproximadas van aparte y sin coordenadas de
   rescate. Mostrar el botón **Marcar aquí**.
5. **Contra la verdad:** `make stop` y después `make verdad` (con Unreal todavía en
   Play). Muestra el error real de cada punto, los duplicados y las falsas alarmas.
6. **Cierre con evidencia:** `docs/resultados/resultados.html` (modelos en VisDrone) y
   [VALIDACION.md](VALIDACION.md) §3 (vuelos SITL).

✏️ Ensayarla al menos 3 veces y cronometrarla: ___ min.

## 2. Plan B

| Si falla | Mostrar en su lugar |
|---|---|
| Unreal, PX4 o AirSim | ✏️ Video grabado de un vuelo completo (FPV y mapa). **Grabarlo antes del 13-11** |
| El vuelo en vivo no deja datos | Un vuelo ya registrado: `python3 scripts/verdad_sitl.py --sin-airsim --vuelo docs/resultados/sitl/20261111_160740` (sin simulador) y abrir su `trail.geojson` + `verdad.geojson` en geojson.io |
| La GPU o ultralytics | `make test` en vivo (lógica sin ROS ni GPU) y `make resultados` (rehace las figuras desde `training/runs/`) |
| Todo | `docs/resultados/resultados.html` abierto offline |

## 3. Preguntas probables

Entiéndanlas con sus palabras; no las memoricen. Durante la defensa no se usa IA.

**¿Por qué tiling?** Al achicar 1920×1080 a 640, una persona lejana queda en 3-4 px y
desaparece. Los tiles de 640 a resolución nativa le conservan los píxeles; el frame
completo sigue viendo a las cercanas (`tiling.py`, docstring).

**¿Por qué fusionan con `ios` y no con IoU?** La misma persona sale entera en el frame
completo y recortada en un tile. `ios` divide por la caja menor y elimina la contenida;
el IoU de las dos es bajo y quedaría duplicada (`tiling.nms`).

**¿Por qué un tracker propio y no `model.track()`?** Las cajas finales salen de fusionar
dos pasadas, y el tracker de ultralytics ve una sola. Además asocia por distancia de
centros normalizada por el tamaño, porque con cajas chicas el IoU cae a 0 con un par de
píxeles de corrimiento (`tracker.py`).

**¿Cómo evitan contar un falso positivo de un instante?** Un track cuenta tras
`min_hits: 3` detecciones, y un waypoint se crea tras seguir a la persona
`min_seen_sec: 2.0` s (`config/sitl.yaml`, `persons.py`).

**¿Cómo evitan contar dos veces a la misma?** Si un track nuevo cae cerca de una persona
ya marcada que no se está viendo, es la misma (`dedupe_radius_m: 6`). Dos personas vistas
en el mismo frame nunca se fusionan (`persons.py`, docstring). Límite: no hay
re-identificación por apariencia.

**¿Cómo ubican a la persona en el suelo?** Centro de la caja → rayo en la cámara →
cuerpo del dron → NED con la actitud de PX4 → corte con un plano a la altura sobre el
suelo (lidar, o sin lidar la altura sobre el despegue) menos 0,9 m → norte/este →
lat/lon. Rayos más rasantes que `max_incidence_deg: 80` no se ubican y quedan
aproximados (`geolocate.locate`).

**¿Qué tan bien ubica?** En el vuelo 4 (código actual, COCO) el error real tuvo una
mediana de 11,4 m (máximo 14,6 m), con un corrimiento medio de +7,0 m al norte
(`docs/resultados/sitl/20261111_160740/informe.txt`).

**¿Es confiable el `±m` que muestra?** No todavía. La persona real cayó dentro de la zona
de 2σ en 0 de 4 puntos ubicados en el vuelo 4, y en 2 de 4 en los vuelos `145432` y
`152146` (mismos `informe.txt`). Si el error informado estuviera bien calibrado, sería
~86 %. Está subestimado: falta calibrar con `scripts/calibrar_geo.py`
([RIESGOS_Y_ETICA.md](RIESGOS_Y_ETICA.md) §1).

**¿Entrenaron el modelo?** Sí: YOLO11 `n` y `s`, partiendo de los pesos COCO, con
VisDrone2019-DET en tiles de 640 y una sola clase `person` ([training/README.md](../training/README.md)).
En VisDrone test-dev (sistema completo, `sitl.yaml`), el recall pasa de 0,119 con COCO a
0,448 (`n_final`) y 0,508 (`s_final`), pero las falsas alarmas por imagen suben de 0,59 a
3,51 y 4,66 (`docs/resultados/resumen.md`).

**¿Por qué `n` y no `s`?** `s` gana ~6 puntos de recall y tarda ~45 % más (36 contra
25 ms por imagen en una RTX 4060, mismo archivo). En la Jetson el tiempo manda. No se
midió en la Jetson.

**¿Funciona en un rescate real?** No está validado. VisDrone es urbano y casi no tiene
personas acostadas; el SITL tiene 3 actores quietos ([RIESGOS_Y_ETICA.md](RIESGOS_Y_ETICA.md) §2,
[VALIDACION.md](VALIDACION.md) §4).

**¿Puede afectar el vuelo?** No: ningún nodo publica en `/fmu/in/*`, solo se lee
`/fmu/out/*`. En la Jetson, Foxglove va en solo lectura.

**¿Y la web de rescate?** No tiene contraseña: cualquiera en la red puede verla.
Desde el 12-11 rechaza los POST de otros sitios (`Origin` y `Content-Type`) y los
`Content-Length` inválidos (`web_map.py`, `test_web_map.py`).

**¿Qué licencia tiene?** Ultralytics es AGPL-3.0. El paquete declara Apache-2.0 y no hay
`LICENSE`: ✏️ decisión del equipo ([RUBRICA.md](RUBRICA.md), [RIESGOS_Y_ETICA.md](RIESGOS_Y_ETICA.md) §6).

**¿Cómo lo reproduzco?** [INSTALACION.md](INSTALACION.md) desde cero, `make doctor`,
`make test`; sin GPU ni Unreal, `make informe-vuelo` y `make resultados`.

**¿Usaron IA?** [DECLARACION_IA.md](DECLARACION_IA.md): qué se hizo con IA, qué se
verificó y qué decidió el equipo.

## 4. Reparto

✏️ Cada integrante debe poder abrir y explicar su parte sin mirar la respuesta.

| Área | Archivos | Responsable |
|---|---|---|
| Problema, alcance, riesgos | README, RIESGOS_Y_ETICA | ✏️ |
| Datos y entrenamiento | `training/`, RESULTADOS | ✏️ |
| Detección: YOLO, tiling, fusión | `inference.py`, `tiling.py` | ✏️ |
| Conteo y waypoints | `tracker.py`, `persons.py` | ✏️ |
| Geolocalización | `geolocate.py`, `gps_node.py` | ✏️ |
| Integración ROS 2, mapa y web | nodos, `trail.py`, `web_map.py`, launch | ✏️ |
| Validación SITL | `verdad_sitl.py`, VALIDACION | ✏️ |

## 5. Antes de entregar

- [ ] Demo ensayada 3 veces y video de respaldo grabado.
- [ ] Lo que se muestra es la misma versión que se entrega (commit: ✏️ ___).
- [ ] Cada cifra mostrada tiene su archivo de origen.
- [ ] Sin datos personales en el repo: el correo del mantenedor se reemplazó, pero sigue
      en el historial de git.
