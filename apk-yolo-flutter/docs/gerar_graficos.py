"""
Gera gráficos comparativos do benchmark de modelos YOLO.

Entrada : docs/benchmark_*.json (usa o mais recente)
Saída   : docs/graficos/*.png

Robusto a:
  - Modelos que rodaram só em GPU ou só em CPU (ignora barra ausente)
  - Modelos com 0 inferências (GPU falhou) — vão como "falha" no speedup,
    pulados no scatter de eficiência
  - Métricas null (avg_ma, total_mah) — pulados nos gráficos correspondentes
  - Número arbitrário de modelos (layout adapta largura automaticamente)
  - Modelos sem rótulo curto definido — usa o label inteiro com quebra de linha
"""

import json
import os
import glob
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(DOCS_DIR, 'graficos')
os.makedirs(OUT_DIR, exist_ok=True)

json_files = sorted(glob.glob(os.path.join(DOCS_DIR, 'benchmark_*.json')))
if not json_files:
    raise FileNotFoundError('Nenhum arquivo benchmark_*.json encontrado em docs/')
json_path = json_files[-1]
print(f'Usando: {os.path.basename(json_path)}')

with open(json_path) as f:
    data = json.load(f)

results = data['results']
duration = data['duration_per_model_sec']

# ---------------------------------------------------------------------------
# Agrupamento por modelo + acelerador
# ---------------------------------------------------------------------------
models_gpu = {r['model_label']: r for r in results if r['use_gpu']}
models_cpu = {r['model_label']: r for r in results if not r['use_gpu']}
labels_all = list(dict.fromkeys(r['model_label'] for r in results))

# Rótulos curtos para os eixos
SHORT_MAP = {
    'EPI YOLO26m fp32': 'EPI-26m\nfp32',
    'EPI YOLO26m fp16': 'EPI-26m\nfp16',
    'yolo26n fp32 (COCO)':       'YOLO26n\nfp32',
    'yolo26n fp16 (COCO)':       'YOLO26n\nfp16',
    'yolo26n int8 DR (COCO)':    'YOLO26n\nint8 DR',
    'yolo26n int8 full (COCO)':  'YOLO26n\nint8 full',
    'yolo11n fp32 (COCO)':       'YOLO11n\nfp32',
    'yolo11n fp16 (COCO)':       'YOLO11n\nfp16',
    'yolo11n int8 DR (COCO)':    'YOLO11n\nint8 DR',
    'yolo11n int8 full (COCO)':  'YOLO11n\nint8 full',
}

def short_of(label):
    if label in SHORT_MAP:
        return SHORT_MAP[label]
    s = re.sub(r'\s*\(COCO\)\s*', '', label).replace(' ', '\n', 1)
    return s

# Paleta
C_GPU = '#1976D2'
C_CPU = '#E53935'
C_FAIL = '#9E9E9E'
ALPHA = 0.85

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.alpha': 0.35,
    'figure.dpi': 150,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def has_inferences(r):
    return r is not None and r.get('total_inferences', 0) > 0

def get_metric(r, key):
    """Devolve o valor da métrica, ou None se ausente / inválido."""
    if not has_inferences(r):
        return None
    v = r.get(key)
    if v is None:
        return None
    return v

def fig_width_for(n_models, per_model=1.1, base=2.5, max_w=20):
    return min(max_w, base + per_model * n_models)

def bar_pair(ax, labels, getter, ylabel, title, fmt='{:.1f}', fail_mark='✗'):
    """Plota barras GPU/CPU lado-a-lado. Marca falhas com hachura cinza."""
    n = len(labels)
    x = np.arange(n)
    w = 0.38

    gpu_vals = [getter(models_gpu.get(l)) for l in labels]
    cpu_vals = [getter(models_cpu.get(l)) for l in labels]

    # Valor máximo para offset das labels
    all_nums = [v for v in gpu_vals + cpu_vals if v is not None]
    max_v = max(all_nums) if all_nums else 1.0

    for i, (gv, cv) in enumerate(zip(gpu_vals, cpu_vals)):
        # GPU
        if gv is not None:
            ax.bar(i - w/2, gv, w, color=C_GPU, alpha=ALPHA,
                   label='GPU' if i == 0 else None)
            ax.text(i - w/2, gv + max_v * 0.012, fmt.format(gv),
                    ha='center', va='bottom', fontsize=7.5,
                    color=C_GPU, fontweight='bold')
        else:
            # Falha: barra cinza pequena com X
            ax.bar(i - w/2, max_v * 0.04, w, color=C_FAIL, alpha=0.4,
                   hatch='///', edgecolor=C_FAIL)
            ax.text(i - w/2, max_v * 0.06, fail_mark,
                    ha='center', va='bottom', fontsize=11, color=C_FAIL)
        # CPU
        if cv is not None:
            ax.bar(i + w/2, cv, w, color=C_CPU, alpha=ALPHA,
                   label='CPU' if i == 0 else None)
            ax.text(i + w/2, cv + max_v * 0.012, fmt.format(cv),
                    ha='center', va='bottom', fontsize=7.5,
                    color=C_CPU, fontweight='bold')
        else:
            ax.bar(i + w/2, max_v * 0.04, w, color=C_FAIL, alpha=0.4,
                   hatch='///', edgecolor=C_FAIL)
            ax.text(i + w/2, max_v * 0.06, fail_mark,
                    ha='center', va='bottom', fontsize=11, color=C_FAIL)

    ax.set_xticks(x)
    ax.set_xticklabels([short_of(l) for l in labels], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=9, loc='upper left')

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {name}')

# Modelos nano (sem o EPI customizado) — usado nos painéis "limpos"
nano_labels = [l for l in labels_all if 'EPI' not in l and 'YOLO26m' not in l]

# ---------------------------------------------------------------------------
# 1. Tempo médio de inferência (ms)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(fig_width_for(len(labels_all)), 5.5))
bar_pair(ax, labels_all, lambda r: get_metric(r, 'avg_inf_ms'),
         'Tempo médio (ms)', 'Tempo médio de inferência — GPU vs CPU',
         fmt='{:.0f}')
save(fig, '01_avg_inf_ms.png')

# ---------------------------------------------------------------------------
# 2. FPS
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(fig_width_for(len(labels_all)), 5.5))
bar_pair(ax, labels_all, lambda r: get_metric(r, 'avg_fps'),
         'FPS', 'FPS efetivo — GPU vs CPU',
         fmt='{:.1f}')
save(fig, '02_fps.png')

# ---------------------------------------------------------------------------
# 3. Corrente média (mA)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(fig_width_for(len(labels_all)), 5.5))
bar_pair(ax, labels_all, lambda r: get_metric(r, 'avg_ma'),
         'Corrente média (mA)', 'Consumo médio de corrente — GPU vs CPU',
         fmt='{:.0f}')
save(fig, '03_avg_ma.png')

# ---------------------------------------------------------------------------
# 4. Energia total (mAh por coleta)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(fig_width_for(len(labels_all)), 5.5))
bar_pair(ax, labels_all, lambda r: get_metric(r, 'total_mah'),
         f'Energia consumida (mAh / {duration}s)',
         'Energia total consumida — GPU vs CPU',
         fmt='{:.2f}')
save(fig, '04_total_mah.png')

# ---------------------------------------------------------------------------
# 5. RAM média (MB)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(fig_width_for(len(labels_all)), 5.5))
bar_pair(ax, labels_all, lambda r: get_metric(r, 'avg_ram_mb'),
         'RAM média (MB)', 'Memória RAM média do processo — GPU vs CPU',
         fmt='{:.0f}')
save(fig, '05_avg_ram_mb.png')

# ---------------------------------------------------------------------------
# 6. Speedup GPU vs CPU
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(fig_width_for(len(labels_all), 1.0), 5.0))
speedups = []
for l in labels_all:
    gv = get_metric(models_gpu.get(l), 'avg_inf_ms')
    cv = get_metric(models_cpu.get(l), 'avg_inf_ms')
    if gv is None or cv is None or gv <= 0:
        speedups.append(None)
    else:
        speedups.append(cv / gv)

x = np.arange(len(labels_all))
max_s = max((s for s in speedups if s is not None), default=1.0)
for i, s in enumerate(speedups):
    if s is None:
        ax.bar(i, max_s * 0.06, color=C_FAIL, alpha=0.4,
               hatch='///', edgecolor=C_FAIL)
        ax.text(i, max_s * 0.08, 'GPU\nfalhou', ha='center', va='bottom',
                fontsize=8, color=C_FAIL, fontweight='bold')
    else:
        color = '#43A047' if s >= 1.05 else ('#FB8C00' if s >= 0.95 else C_CPU)
        ax.bar(i, s, color=color, alpha=ALPHA)
        ax.text(i, s + max_s * 0.015, f'{s:.2f}×',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([short_of(l) for l in labels_all], fontsize=8)
ax.set_ylabel('Speedup (CPU ms ÷ GPU ms)', fontsize=10)
ax.set_title('Ganho de velocidade da GPU sobre CPU', fontsize=12,
             fontweight='bold', pad=10)
save(fig, '06_gpu_speedup.png')

# ---------------------------------------------------------------------------
# 7. Mapa eficiência: FPS vs mAh (scatter)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 7))
for l in labels_all:
    rg = models_gpu.get(l)
    rc = models_cpu.get(l)
    sl = short_of(l).replace('\n', ' ')

    if has_inferences(rg) and rg.get('total_mah') is not None:
        ax.scatter(rg['total_mah'], rg['avg_fps'], color=C_GPU, s=130, zorder=3)
        ax.annotate(f'{sl} GPU',
                    (rg['total_mah'], rg['avg_fps']),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=7.5, color=C_GPU)
    if has_inferences(rc) and rc.get('total_mah') is not None:
        ax.scatter(rc['total_mah'], rc['avg_fps'], color=C_CPU, s=130, zorder=3)
        ax.annotate(f'{sl} CPU',
                    (rc['total_mah'], rc['avg_fps']),
                    textcoords='offset points', xytext=(6, -10),
                    fontsize=7.5, color=C_CPU)
ax.set_xlabel(f'Energia consumida (mAh / {duration}s) — menor é melhor →',
              fontsize=10)
ax.set_ylabel('FPS — maior é melhor ↑', fontsize=10)
ax.set_title('Mapa de eficiência: FPS × Energia consumida',
             fontsize=12, fontweight='bold', pad=10)
ax.legend(handles=[mpatches.Patch(color=C_GPU, label='GPU'),
                   mpatches.Patch(color=C_CPU, label='CPU')], fontsize=9)
save(fig, '07_fps_vs_mah.png')

# ---------------------------------------------------------------------------
# 8. Painel resumo (somente modelos nano)
# ---------------------------------------------------------------------------
if nano_labels:
    fig, axes = plt.subplots(1, 3, figsize=(fig_width_for(len(nano_labels), 1.4, base=4), 5.5))
    fig.suptitle('Modelos nano — comparativo resumido', fontsize=13, fontweight='bold')
    metrics = [
        ('avg_inf_ms', 'Avg ms', '{:.0f}'),
        ('avg_fps',    'FPS',    '{:.1f}'),
        ('total_mah',  'mAh',    '{:.2f}'),
    ]
    for ax, (key, label, fmt) in zip(axes, metrics):
        bar_pair(ax, nano_labels, lambda r, k=key: get_metric(r, k),
                 label, label, fmt=fmt)
    fig.tight_layout()
    save(fig, '08_nano_resumo.png')

# ---------------------------------------------------------------------------
# 9. Best-of: melhor configuração por modelo (escolhe GPU ou CPU, o que for mais rápido)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(fig_width_for(len(labels_all), 1.0), 5.5))
best_ms, best_label, best_color = [], [], []
for l in labels_all:
    gv = get_metric(models_gpu.get(l), 'avg_inf_ms')
    cv = get_metric(models_cpu.get(l), 'avg_inf_ms')
    if gv is None and cv is None:
        best_ms.append(0); best_label.append('—'); best_color.append(C_FAIL)
    elif gv is None:
        best_ms.append(cv); best_label.append('CPU'); best_color.append(C_CPU)
    elif cv is None:
        best_ms.append(gv); best_label.append('GPU'); best_color.append(C_GPU)
    elif gv < cv:
        best_ms.append(gv); best_label.append('GPU'); best_color.append(C_GPU)
    else:
        best_ms.append(cv); best_label.append('CPU'); best_color.append(C_CPU)

x = np.arange(len(labels_all))
ax.bar(x, best_ms, color=best_color, alpha=ALPHA)
for i, (v, lab) in enumerate(zip(best_ms, best_label)):
    if v > 0:
        ax.text(i, v + max(best_ms) * 0.015, f'{v:.0f}ms\n({lab})',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([short_of(l) for l in labels_all], fontsize=8)
ax.set_ylabel('Melhor tempo de inferência (ms)', fontsize=10)
ax.set_title('Melhor configuração por modelo (menor latência)',
             fontsize=12, fontweight='bold', pad=10)
ax.legend(handles=[mpatches.Patch(color=C_GPU, label='GPU foi melhor'),
                   mpatches.Patch(color=C_CPU, label='CPU foi melhor')],
          fontsize=9)
save(fig, '09_best_per_model.png')

print('\nPronto! Gráficos salvos em docs/graficos/')
