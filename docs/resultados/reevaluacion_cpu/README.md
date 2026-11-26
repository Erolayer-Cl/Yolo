# Reevaluación en CPU (26-11-2026)

Nivel 2 (sistema completo, `sitl.yaml`, IoU 0.5) repetido sobre las 1610 imágenes de
VisDrone2019-DET test-dev (copia de Ultralytics), en CPU, con Ultralytics 8.4.208:

```bash
python training/extra_reevaluar_cpu.py --src <VisDrone> --nombre n_final --pesos models/visdrone_n.pt --out docs/resultados/reevaluacion_cpu
python training/extra_reevaluar_cpu.py --src <VisDrone> --nombre coco    --pesos yolo11n.pt          --out docs/resultados/reevaluacion_cpu
```

`visdrone_n.pt` es el `best.pt` de `n_final` (SHA-256
`dda24445368c99e99ab78e4acc9aa24fca5aaae083f9da647becfd481facdb68`, `models/visdrone_n.pt`).

| Modelo | Precisión | Recall | Recall <16 px | Falsas alarmas/img | Original (RTX 4060, `resumen.json`) |
|---|---|---|---|---|---|
| n_final | 0.685 | 0.449 | 0.374 | 3.51 | 0.685 / 0.448 / 0.373 / 3.51 |
| coco | 0.773 | 0.119 | 0.050 | 0.60 | 0.775 / 0.119 / 0.050 / 0.59 |

Las diferencias vienen de calcular en CPU con precisión completa (la original usó fp16 en
GPU). El número de personas por tamaño es idéntico. Los ms/img de estos JSON son de CPU
y no sirven para estimar la Jetson.
