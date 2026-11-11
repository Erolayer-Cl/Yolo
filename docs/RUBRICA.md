# Rúbrica → evidencia

Dónde está, en este repo, la evidencia de cada criterio de la rúbrica del Proyecto
Integrador (UBO DSI 2026_S2, v02). Sirve para revisar la entrega y para preparar la
defensa.

Estado al 2026-11-11:
- ✅ = hay evidencia.
- ⚠️ = parcial: se dice qué falta.
- ✏️ = lo tiene que hacer o escribir el equipo.

Regla transversal de la rúbrica: solo resultados de ejecuciones reales, y uso de IA
declarado. Por eso, cada cifra de `docs/` apunta al archivo que la generó.

## EVAL-01: prototipo funcional (40 %, 13-11-2026)

| # | Criterio (pts) | Evidencia | Estado | Falta |
|---|---|---|---|---|
| 1 | Problema y alcance (10) | [README: Problema, usuarios y alcance](../README.md#problema-usuarios-y-alcance) | ✅ | — |
| 2 | Datos y trazabilidad (15) | [training/README: Datos y licencias](../training/README.md#datos-y-licencias) (origen, licencia, preparación, estadísticas), [Decisiones](../training/README.md#decisiones), [Limitaciones](../training/README.md#limitaciones); `e1_preparar_datos.py` | ✅ | — |
| 3 | Técnica/modelo y decisión (15) | Comparación de COCO vs `n_final` vs `s_final` en dos niveles: [RESULTADOS.md](resultados/RESULTADOS.md), [training/README: Resultados](../training/README.md#resultados-visdrone-test-dev-1610-imágenes) (decisión y supuestos) | ⚠️ | Sin tuning de hiperparámetros ni varias semillas (declarado). Sin medir ms en la Jetson |
| 4 | Arquitectura e integración (20) | [README: Arquitectura](../README.md#arquitectura) (diagrama y tópicos), SITL completo con `make sitl` | ✅ | — |
| 5 | Validación y resultados (20) | [VALIDACION.md](VALIDACION.md): tests, VisDrone en dos niveles, SITL contra verdad de terreno, análisis de errores y lo no validado; [resultados/](resultados/) | ⚠️ | Vuelo con el código actual hecho (vuelo 4, COCO). Falta: un vuelo con rosbag y `MODEL=visdrone_n.pt`, y correr `calibrar_geo.py` sobre ese rosbag |
| 6 | Reproducibilidad y demostración (15) | [INSTALACION.md](INSTALACION.md) desde cero, `make doctor`, `make` (atajos), [requirements.txt](../training/requirements.txt), `sim/airsim/` versionado, `make informe-vuelo` y `make resultados` rehacen los informes sin GPU ni Unreal | ⚠️ ✏️ | Proyecto de Unreal documentado (`Pruebas`, INSTALACION 5a). Falta confirmar cómo se compiló AirLib (5b) y ensayar la demo |
| 7 | Responsabilidad, seguridad e IA (5) | [RIESGOS_Y_ETICA.md](RIESGOS_Y_ETICA.md), [DECLARACION_IA.md](DECLARACION_IA.md) | ⚠️ ✏️ | Completar las ✏️ de DECLARACION_IA. Corregir la seguridad de la web (§4 de RIESGOS) |

## EVAL-02: informe y defensa (30 %, 27-11-2026)

El informe es [INFORME.md](INFORME.md) (esqueleto con las 8 secciones en este orden).

| # | Criterio (pts) | De dónde sale | Estado | Falta |
|---|---|---|---|---|
| 1 | Contexto, problema y objetivos (10) | README: Problema y alcance | ⚠️ ✏️ | Redactar el contexto (búsqueda y rescate en Chile, por qué un dron) |
| 2 | Metodología, datos y arquitectura (20) | training/README, README: Arquitectura, `sim/airsim/README.md` | ⚠️ ✏️ | Pasarlo a prosa en el informe, con las decisiones y su porqué |
| 3 | Resultados, validación y análisis crítico (20) | RESULTADOS.md (figuras), VALIDACION.md | ⚠️ | Nuevo vuelo SITL; conectar cada resultado con un objetivo |
| 4 | Ética, seguridad, impacto y limitaciones (10) | RIESGOS_Y_ETICA.md | ✅ | — |
| 5 | Calidad del informe y fuentes (15) | INFORME.md § Referencias | ⚠️ ✏️ | Redacción final; citar en el texto |
| 6 | Presentación técnica (10) | Figuras de `resultados/img/`, `resultados.html` | ✏️ | Hacer las diapositivas |
| 7 | Defensa y dominio del equipo (10) | DECLARACION_IA: qué puede explicar cada integrante | ✏️ | Repartir y ensayar: cada integrante explica una parte |
| 8 | Integridad y declaración de IA (5) | DECLARACION_IA.md, trailers `Co-Authored-By` en git | ⚠️ ✏️ | Completar las ✏️ |

## Antes del 13-11 (orden sugerido)

1. Completar las ✏️ de [DECLARACION_IA.md](DECLARACION_IA.md) y de
   [INSTALACION.md](INSTALACION.md) (paso 5b).
2. Volar el SITL con `make sitl MODEL=visdrone_n.pt` a 20–40 m. Después correr
   `make verdad`, copiar el vuelo a `resultados/sitl/` y actualizar VALIDACION §3.
3. Ensayar la demo de punta a punta: Unreal en Play → `make sitl` → volar → mapa `:8081` →
   `make stop` → `make verdad`. Mostrar `resultados.html` para las métricas.
4. Si queda tiempo: las correcciones de la revisión de código, la de la web primero.
