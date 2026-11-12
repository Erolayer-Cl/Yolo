## Nivel 1 — modelo (tiles de test)

| Modelo | Precisión | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| coco | 0.416 | 0.197 | 0.184 | 0.077 |
| n_final | 0.656 | 0.427 | 0.463 | 0.187 |
| s_final | 0.686 | 0.469 | 0.508 | 0.211 |

## Nivel 2 — sistema con sitl.yaml (imágenes completas, conf 0.35 / tiles 0.25, overlap 0.1, IoU 0.5)

| Modelo | Precisión | Recall | F1 | Recall <16px | Recall 16-32px | Recall >=32px | Falsas alarmas/img | ms/img |
|---|---|---|---|---|---|---|---|---|
| coco | 0.775 | 0.119 | 0.207 | 0.050 | 0.335 | 0.669 | 0.59 | 24 |
| n_final | 0.685 | 0.448 | 0.542 | 0.373 | 0.741 | 0.791 | 3.51 | 25 |
| s_final | 0.650 | 0.508 | 0.570 | 0.437 | 0.782 | 0.845 | 4.66 | 36 |

## Nivel 2 — sistema con jetson.yaml (imágenes completas, conf 0.35 / tiles 0.25, overlap 0.15, IoU 0.5)

| Modelo | Precisión | Recall | F1 | Recall <16px | Recall 16-32px | Recall >=32px | Falsas alarmas/img | ms/img |
|---|---|---|---|---|---|---|---|---|
| coco | 0.774 | 0.120 | 0.207 | 0.050 | 0.335 | 0.673 | 0.60 | 25 |
| n_final | 0.685 | 0.448 | 0.542 | 0.374 | 0.740 | 0.793 | 3.51 | 25 |
| s_final | 0.651 | 0.510 | 0.572 | 0.439 | 0.783 | 0.846 | 4.65 | 36 |

Personas en test por tamaño: {'<16px': 21956, '16-32px': 4339, '>=32px': 1087} (1610 imágenes).
Nivel 1: precisión y recall al umbral que maximiza F1 (Ultralytics). Nivel 2: a los umbrales reales del detector. ms medidos en esta GPU, no en la Jetson. Los tiles corren en todas las imágenes (en vuelo, cada every_n_frames).
