#!/usr/bin/env python3
"""Etapa 2: tuning manual, cambiando UN hiperparámetro a la vez.

  A  valores por defecto                   (base)
  B  scale 0.5 -> 0.25                     achicar menos: las personas chicas no desaparecen
  C  flipud 0 -> 0.5                       desde arriba no hay "arriba": más variedad gratis
  D  B + C                                 solo si B y C ayudaron cada uno

Todas con yolo11n, pocas épocas, misma semilla y los mismos datos (train.py),
medidas en VAL. test no se toca: si se eligieran hiperparámetros mirando test,
la evaluación final dejaría de ser honesta.

Criterio "ayuda": sube el recall al menos --min-mejora respecto de A y el mAP50
no baja más de --tolerancia. Diferencias más chicas que eso pueden ser ruido.

Cada prueba es un entrenamiento de horas: si el script se corta, al volver a
correrlo salta las pruebas ya terminadas (quedan en resultados.json).

Uso:
  python tune.py --data ~/datasets/visdrone_tiles/data.yaml
Salida (en runs/tuning/):
  resultados.json  métricas de cada prueba
  tabla.md         tabla comparativa para el informe
  ganador.json     hiperparámetros elegidos -> train.py --cambios
"""
import argparse
import json
from pathlib import Path

from train import COMUN, entrenar

PRUEBAS = {
    'A': {},
    'B': {'scale': 0.25},
    'C': {'flipud': 0.5},
}
DESCRIPCION = {
    'A': 'valores por defecto',
    'B': 'scale 0.5 -> 0.25',
    'C': 'flipud 0 -> 0.5',
    'D': 'B + C',
}


def ayuda(m, base, min_mejora, tolerancia):
    """¿La prueba `m` mejora el recall sin hundir el mAP50 respecto de la base?"""
    return (m['recall'] - base['recall'] >= min_mejora
            and base['mAP50'] - m['mAP50'] <= tolerancia)


def correr(nombre, cambios, a, resultados, archivo):
    if nombre in resultados:
        print(f'Prueba {nombre} ya hecha, se salta: {resultados[nombre]["metricas"]}')
        return
    print(f'\n=== Prueba {nombre}: {DESCRIPCION[nombre]} ===')
    # exist_ok: si una prueba se cortó a medias, se rehace en la misma carpeta.
    extra = {'exist_ok': True}
    if a.fraction < 1.0:
        extra['fraction'] = a.fraction
    metricas, run_dir = entrenar('yolo11n.pt', a.data, f'prueba_{nombre}', a.epochs,
                                 cambios, a.project, **extra)
    resultados[nombre] = {'cambios': cambios, 'metricas': metricas, 'run': str(run_dir)}
    archivo.write_text(json.dumps(resultados, indent=2))      # guardar tras cada prueba


def tabla(resultados):
    base = resultados['A']['metricas']
    filas = ['| Prueba | Cambio | Precisión | Recall | mAP50 | mAP50-95 | Δ recall vs A |',
             '|---|---|---|---|---|---|---|']
    for k, r in resultados.items():
        m = r['metricas']
        d = m['recall'] - base['recall']
        filas.append(f'| {k} | {DESCRIPCION[k]} | {m["precision"]:.3f} | {m["recall"]:.3f} '
                     f'| {m["mAP50"]:.3f} | {m["mAP50-95"]:.3f} | {d:+.3f} |')
    return '\n'.join(filas)


def main():
    ap = argparse.ArgumentParser(description='Tuning manual (etapa 2).')
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--min-mejora', type=float, default=0.005, help='mejora mínima de recall')
    ap.add_argument('--tolerancia', type=float, default=0.01, help='caída máxima de mAP50')
    ap.add_argument('--fraction', type=float, default=1.0,
                    help='fracción de train a usar (solo para probar el script)')
    ap.add_argument('--batch', type=int, default=COMUN['batch'],
                    help='bajalo si la GPU se queda sin memoria (igual en todas las pruebas)')
    ap.add_argument('--project', default='runs/tuning')
    a = ap.parse_args()
    a.data = a.data.expanduser()
    COMUN['batch'] = a.batch

    out = Path(a.project)
    out.mkdir(parents=True, exist_ok=True)
    archivo = out / 'resultados.json'
    resultados = json.loads(archivo.read_text()) if archivo.exists() else {}

    for nombre, cambios in PRUEBAS.items():
        correr(nombre, cambios, a, resultados, archivo)

    base = resultados['A']['metricas']
    utiles = [k for k in ('B', 'C')
              if ayuda(resultados[k]['metricas'], base, a.min_mejora, a.tolerancia)]
    if len(utiles) == 2:                       # D solo tiene sentido si ambas ayudaron
        correr('D', {**PRUEBAS['B'], **PRUEBAS['C']}, a, resultados, archivo)
    else:
        print('\nD no se corre: B y C no ayudaron las dos.')

    # Ganador: entre las que ayudan, la de mayor recall. Si ninguna, A (por defecto).
    candidatas = [k for k in resultados
                  if k != 'A' and ayuda(resultados[k]['metricas'], base,
                                        a.min_mejora, a.tolerancia)]
    ganador = max(candidatas, key=lambda k: resultados[k]['metricas']['recall'],
                  default='A')

    t = tabla(resultados)
    nota = (f'\n\nGanador: **{ganador}** ({DESCRIPCION[ganador]}). Criterio: recall '
            f'+{a.min_mejora} o más sobre A y mAP50 a no más de {a.tolerancia} de A. '
            f'{a.epochs} épocas por prueba, yolo11n, medido en val.')
    (out / 'tabla.md').write_text(t + nota + '\n')
    (out / 'ganador.json').write_text(json.dumps(
        {'prueba': ganador, 'cambios': resultados[ganador]['cambios']}, indent=2))
    print('\n' + t + nota)


if __name__ == '__main__':
    main()
