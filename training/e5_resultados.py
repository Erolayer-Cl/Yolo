#!/usr/bin/env python3
"""Etapa 5: gráficos, tabla y página con los resultados de los modelos.

No corre ningún modelo: solo LEE lo que dejaron las etapas 2 y 3, así cada cifra
sale de una ejecución real.
  runs/eval/resumen.json               métricas de test-dev (e3_evaluar.py)
  runs/train/<run>/results.csv         métricas de val y pérdidas por época (Ultralytics)
  runs/train/<run>/ficha_modelo.json   datos, entorno y métricas de val (e2_entrenar.py)

Escribe en docs/resultados/ (versionado, para el informe y la presentación):
  RESULTADOS.md          tablas + gráficos + de dónde sale cada número
  resultados.html        lo mismo en una sola página, sin internet (doble clic)
  img/comparacion_metricas.png, img/recall_por_tamano.png, img/costo.png,
  img/curvas_entrenamiento.png
  resumen.md, resumen.json, ficha_<run>.json   copias de los originales

Uso (desde training/):
  python e5_resultados.py
  python e5_resultados.py --runs runs --out ../docs/resultados
"""
import argparse
import base64
import csv
import html
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

AQUI = Path(__file__).resolve().parent
TAMANOS = ('<16px', '16-32px', '>=32px')
COLORES = {'coco': '#8c8c8c', 'n_final': '#2a6fdb', 's_final': '#e07b00'}
OTROS = ['#2ca02c', '#9467bd', '#d62728', '#17becf']

GLOSARIO = [
    ('Precisión', 'De las detecciones, cuántas eran personas de verdad. Baja = muchas falsas alarmas.'),
    ('Recall', 'De las personas etiquetadas, cuántas se encontraron. Baja = personas perdidas.'),
    ('F1', 'Media armónica de precisión y recall: un solo número que pide las dos.'),
    ('mAP50', 'Promedio de la precisión en todos los umbrales de score, contando como acierto '
              'una caja con IoU ≥ 0.5 con la etiquetada. Estándar de VisDrone.'),
    ('mAP50-95', 'Lo mismo promediando IoU de 0.5 a 0.95: exige cajas más justas.'),
    ('Falsas alarmas/img', 'Detecciones sin persona etiquetada, por imagen de test-dev.'),
    ('ms/img', 'Tiempo del sistema completo por imagen en la GPU de evaluación (RTX 4060), no en la Jetson.'),
]


def parse_args():
    ap = argparse.ArgumentParser(description='Etapa 5: gráficos y página de resultados.')
    ap.add_argument('--runs', type=Path, default=AQUI / 'runs')
    ap.add_argument('--out', type=Path, default=AQUI.parent / 'docs' / 'resultados')
    ap.add_argument('--config', default='sitl', help='config del nivel 2 para los gráficos')
    return ap.parse_args()


def color(nombre, i):
    return COLORES.get(nombre, OTROS[i % len(OTROS)])


def leer_csv(path: Path):
    """results.csv de Ultralytics -> {columna: [valores]} (las columnas pueden traer espacios)."""
    with open(path) as f:
        filas = list(csv.DictReader(f))
    cols = {}
    for fila in filas:
        for k, v in fila.items():
            cols.setdefault(k.strip(), []).append(float(v))
    return cols


# --------------------------------------------------------------------- gráficos
def barras(ax, grupos, modelos, valores, titulo, fmt='{:.3f}', ylim=None):
    """Barras agrupadas: un grupo por métrica, una barra por modelo."""
    n = len(modelos)
    ancho = 0.8 / n
    for i, m in enumerate(modelos):
        xs = [g + (i - (n - 1) / 2) * ancho for g in range(len(grupos))]
        b = ax.bar(xs, valores[m], ancho, label=m, color=color(m, i))
        ax.bar_label(b, labels=[fmt.format(v) for v in valores[m]], fontsize=7, padding=2)
    ax.set_xticks(range(len(grupos)), grupos)
    ax.set_title(titulo, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)


def graficos(res, curvas, cfg, img: Path):
    img.mkdir(parents=True, exist_ok=True)
    modelos = list(res)
    out = {}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    g1 = ['precision', 'recall', 'mAP50', 'mAP50-95']
    barras(a1, ['Precisión', 'Recall', 'mAP50', 'mAP50-95'], modelos,
           {m: [res[m]['tiles'][k] for k in g1] for m in modelos},
           'Nivel 1: el modelo solo, sobre tiles de 640 (test-dev)', ylim=(0, 1))
    g2 = ['precision', 'recall', 'F1']
    barras(a2, ['Precisión', 'Recall', 'F1'], modelos,
           {m: [res[m]['sistema'][cfg][k] for k in g2] for m in modelos},
           f'Nivel 2: sistema completo con {cfg}.yaml (imágenes enteras)', ylim=(0, 1))
    a1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out['comparacion_metricas'] = guardar(fig, img / 'comparacion_metricas.png')

    fig, ax = plt.subplots(figsize=(7, 4))
    barras(ax, ['< 16 px', '16–32 px', '≥ 32 px'], modelos,
           {m: [res[m]['sistema'][cfg]['recall_por_tamano'][t] for t in TAMANOS] for m in modelos},
           f'Recall por tamaño de persona (lado menor de la caja, {cfg}.yaml)', ylim=(0, 1))
    personas = res[modelos[0]]['sistema'][cfg]['personas_por_tamano']
    ax.set_xlabel('personas en test-dev: ' + ', '.join(f'{t}: {personas[t]}' for t in TAMANOS),
                  fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out['recall_por_tamano'] = guardar(fig, img / 'recall_por_tamano.png')

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    barras(a1, [''], modelos, {m: [res[m]['sistema'][cfg]['falsas_alarmas_por_imagen']]
                               for m in modelos}, 'Falsas alarmas por imagen', fmt='{:.2f}')
    barras(a2, [''], modelos, {m: [res[m]['sistema'][cfg]['ms_por_imagen']] for m in modelos},
           'ms por imagen (RTX 4060, sistema completo)', fmt='{:.0f}')
    a1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out['costo'] = guardar(fig, img / 'costo.png')

    if curvas:
        paneles = [('metrics/mAP50(B)', 'mAP50 (val)'), ('metrics/mAP50-95(B)', 'mAP50-95 (val)'),
                   ('metrics/precision(B)', 'Precisión (val)'), ('metrics/recall(B)', 'Recall (val)'),
                   ('box_loss', 'Pérdida de caja (— train, -- val)'),
                   ('cls_loss', 'Pérdida de clase (— train, -- val)')]
        fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
        for ax, (col, titulo) in zip(axs.flat, paneles):
            for i, (m, c) in enumerate(curvas.items()):
                col_c = color(m, i)
                if col.endswith('_loss'):
                    ax.plot(c['epoch'], c[f'train/{col}'], color=col_c, label=f'{m} train')
                    ax.plot(c['epoch'], c[f'val/{col}'], color=col_c, ls='--', label=f'{m} val')
                else:
                    ax.plot(c['epoch'], c[col], color=col_c, label=m)
            ax.set_title(titulo, fontsize=10)
            ax.set_xlabel('época', fontsize=8)
            ax.grid(alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)
        axs.flat[0].legend(frameon=False, fontsize=8)
        fig.suptitle('Entrenamiento por época (validación VisDrone val; de results.csv)', fontsize=11)
        fig.tight_layout()
        out['curvas_entrenamiento'] = guardar(fig, img / 'curvas_entrenamiento.png')
    return out


def guardar(fig, path: Path) -> Path:
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------- tablas
def tablas(res, curvas, fichas):
    """Filas (encabezado, filas) de cada tabla, compartidas por el .md y el .html."""
    modelos = list(res)
    t1 = (['Modelo', 'Pesos', 'Precisión', 'Recall', 'mAP50', 'mAP50-95'],
          [[m, res[m]['pesos']] + [f"{res[m]['tiles'][k]:.3f}"
                                   for k in ('precision', 'recall', 'mAP50', 'mAP50-95')]
           for m in modelos])
    t2 = {}
    for cfg in res[modelos[0]]['sistema']:
        filas = []
        for m in modelos:
            s = res[m]['sistema'][cfg]
            filas.append([m, f"{s['precision']:.3f}", f"{s['recall']:.3f}", f"{s['F1']:.3f}"]
                         + [f"{s['recall_por_tamano'][t]:.3f}" for t in TAMANOS]
                         + [f"{s['falsas_alarmas_por_imagen']:.2f}", f"{s['ms_por_imagen']:.0f}",
                            str(s['imagenes'])])
        t2[cfg] = (['Modelo', 'Precisión', 'Recall', 'F1', 'Recall <16 px', 'Recall 16–32 px',
                    'Recall ≥32 px', 'Falsas alarmas/img', 'ms/img', 'Imágenes'], filas)
    t3 = None
    if curvas:
        filas = []
        for m, c in curvas.items():
            mejor = max(range(len(c['epoch'])), key=lambda i: c['metrics/mAP50-95(B)'][i])
            f = fichas.get(m, {})
            ent = f.get('entorno', {})
            filas.append([m, f.get('modelo_base', '?'), str(int(c['epoch'][-1])),
                          str(int(c['epoch'][mejor])),
                          f"{c['metrics/mAP50(B)'][mejor]:.3f}",
                          f"{c['metrics/mAP50-95(B)'][mejor]:.3f}",
                          _horas(c),
                          f"{ent.get('gpu', '?')}, ultralytics {ent.get('ultralytics', '?')}"])
        t3 = (['Run', 'Base', 'Épocas', 'Mejor época (mAP50-95 val)', 'mAP50 val', 'mAP50-95 val',
               'Horas', 'Entorno'], filas)
    return t1, t2, t3


def _horas(c):
    """`time` de Ultralytics es acumulado (s desde el inicio): la última fila es el total."""
    return f"{c['time'][-1] / 3600:.1f}" if 'time' in c else '?'


def md_tabla(enc, filas):
    lineas = ['| ' + ' | '.join(enc) + ' |', '|' + '---|' * len(enc)]
    lineas += ['| ' + ' | '.join(f) + ' |' for f in filas]
    return '\n'.join(lineas)


def html_tabla(enc, filas):
    cab = ''.join(f'<th>{html.escape(e)}</th>' for e in enc)
    cuerpo = ''.join('<tr>' + ''.join(f'<td>{html.escape(c)}</td>' for c in f) + '</tr>'
                     for f in filas)
    return f'<div class="t"><table><thead><tr>{cab}</tr></thead><tbody>{cuerpo}</tbody></table></div>'


# ------------------------------------------------------------------- documentos
FIGURAS = [
    ('comparacion_metricas', 'Métricas de los modelos en los dos niveles'),
    ('recall_por_tamano', 'Recall según el tamaño de la persona en la imagen'),
    ('costo', 'Costo: falsas alarmas y tiempo'),
    ('curvas_entrenamiento', 'Curvas de entrenamiento de n_final y s_final'),
]
ORIGEN = ('Cifras de `training/runs/eval/resumen.json` (salida de `e3_evaluar.py` sobre '
          'VisDrone2019-DET test-dev, 1610 imágenes) y de `training/runs/train/<run>/results.csv` '
          'y `ficha_modelo.json` (salida de `e2_entrenar.py`). Generado por '
          '`training/e5_resultados.py`: no hay números escritos a mano.')


def escribir_md(out: Path, t1, t2, t3, figs):
    p = ['# Resultados de los modelos', '', ORIGEN, '',
         'En detección de objetos no se usa *accuracy* (no hay "negativos verdaderos" que '
         'contar: el fondo es infinito). Lo equivalente es precisión, recall, F1 y mAP; '
         'ver el glosario al final.', '',
         '## Nivel 1: el modelo solo (tiles de 640, Ultralytics val)', '', md_tabla(*t1), '']
    for cfg, t in t2.items():
        p += [f'## Nivel 2: sistema completo con `{cfg}.yaml`', '',
              'Imágenes enteras por `detect_persons()` (frame completo + tiles + fusión) a los '
              'umbrales reales del detector.', '', md_tabla(*t), '']
    if t3:
        p += ['## Entrenamiento', '', md_tabla(*t3), '',
              'El mAP de val es más alto que el de test-dev: val se usó para quedarse con la '
              'mejor época (`best.pt`), y test-dev tiene más personas chicas (83 % de menos de '
              '16 px, contra 70 % en val; tabla de `training/README.md`). La cifra que vale para '
              'comparar es la de test-dev.', '']
    p += ['## Gráficos', '']
    for clave, titulo in FIGURAS:
        if clave in figs:
            p += [f'### {titulo}', '', f'![{titulo}](img/{figs[clave].name})', '']
    p += ['## Glosario', ''] + [f'- **{k}**: {v}' for k, v in GLOSARIO] + [
        '', 'Lectura y decisión: [`training/README.md`](../../training/README.md#resultados-visdrone-test-dev-1610-imágenes). '
        'Límites de esta evaluación: [`docs/VALIDACION.md`](../VALIDACION.md).', '']
    (out / 'RESULTADOS.md').write_text('\n'.join(p), encoding='utf-8')


TITULO_HTML = 'Métricas del detector de personas'
CSS = """
:root{--bg:#fff;--fg:#1b1f24;--mut:#5b636d;--line:#d9dde3;--card:#f6f8fa;--acc:#2a6fdb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#111418;--fg:#e6e8eb;--mut:#9aa3ad;--line:#2c3239;--card:#181c21;--acc:#6ea3ff;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#111418;--fg:#e6e8eb;--mut:#9aa3ad;--line:#2c3239;--card:#181c21;--acc:#6ea3ff;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding-block:24px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h1{font-size:1.7rem;margin:.2em 0;text-wrap:balance}
h2{font-size:1.2rem;margin:2em 0 .6em;border-bottom:1px solid var(--line);padding-bottom:.3em}
p.mut,li{color:var(--mut)}.t{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.9rem}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
thead th{background:var(--card)}figure{margin:16px 0;background:#fff;border:1px solid var(--line);border-radius:8px;padding:8px}
figure img{width:100%;height:auto;display:block}figcaption{color:#5b636d;font-size:.85rem;padding:4px}
dt{font-weight:600}dd{margin:0 0 .6em;color:var(--mut)}code{background:var(--card);padding:1px 4px;border-radius:4px}
"""


def escribir_html(out: Path, t1, t2, t3, figs):
    def fig(clave, titulo):
        if clave not in figs:
            return ''
        b64 = base64.b64encode(figs[clave].read_bytes()).decode()
        return (f'<figure><img alt="{html.escape(titulo)}" src="data:image/png;base64,{b64}">'
                f'<figcaption>{html.escape(titulo)}</figcaption></figure>')

    origen = html.escape(ORIGEN.replace('`', ''))
    partes = [f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
              f'<meta name="viewport" content="width=device-width,initial-scale=1">'
              f'<title>{TITULO_HTML}</title><meta name="claude:viewport-fit-sides" content="cover">'
              f'<style>{CSS}</style></head><body><main>',
              '<h1>Detección de personas desde dron: resultados</h1>',
              f'<p class="mut">{origen}</p>',
              '<p class="mut">En detección no se usa <i>accuracy</i>: lo equivalente es precisión, '
              'recall, F1 y mAP (glosario al final).</p>',
              '<h2>Nivel 1: el modelo solo (tiles de 640)</h2>', html_tabla(*t1),
              fig('comparacion_metricas', FIGURAS[0][1])]
    for cfg, t in t2.items():
        partes += [f'<h2>Nivel 2: sistema completo con {html.escape(cfg)}.yaml</h2>', html_tabla(*t)]
    partes += [fig('recall_por_tamano', FIGURAS[1][1]), fig('costo', FIGURAS[2][1])]
    if t3:
        partes += ['<h2>Entrenamiento</h2>', html_tabla(*t3), fig('curvas_entrenamiento', FIGURAS[3][1])]
    partes += ['<h2>Glosario</h2><dl>']
    partes += [f'<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>' for k, v in GLOSARIO]
    partes += ['</dl></main></body></html>']
    (out / 'resultados.html').write_text('\n'.join(partes), encoding='utf-8')


def main():
    a = parse_args()
    resumen = a.runs / 'eval' / 'resumen.json'
    if not resumen.exists():
        raise SystemExit(f'No existe {resumen}: correr antes e3_evaluar.py (etapa 3).')
    res = json.loads(resumen.read_text())
    cfg = a.config if a.config in next(iter(res.values()))['sistema'] else \
        next(iter(next(iter(res.values()))['sistema']))
    a.out.mkdir(parents=True, exist_ok=True)

    curvas, fichas = {}, {}
    for m in res:
        run = a.runs / 'train' / m
        if (run / 'results.csv').exists():
            curvas[m] = leer_csv(run / 'results.csv')
        if (run / 'ficha_modelo.json').exists():
            fichas[m] = json.loads((run / 'ficha_modelo.json').read_text())
            shutil.copy2(run / 'ficha_modelo.json', a.out / f'ficha_{m}.json')
    for nombre in ('resumen.json', 'resumen.md'):
        if (a.runs / 'eval' / nombre).exists():
            shutil.copy2(a.runs / 'eval' / nombre, a.out / nombre)

    figs = graficos(res, curvas, cfg, a.out / 'img')
    t1, t2, t3 = tablas(res, curvas, fichas)
    escribir_md(a.out, t1, t2, t3, figs)
    escribir_html(a.out, t1, t2, t3, figs)
    print(f'Modelos: {", ".join(res)} | curvas: {", ".join(curvas) or "ninguna"}')
    for p in [a.out / 'RESULTADOS.md', a.out / 'resultados.html', *figs.values()]:
        print('  ', p)


if __name__ == '__main__':
    main()
