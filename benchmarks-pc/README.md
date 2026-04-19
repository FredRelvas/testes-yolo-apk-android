# benchmarks-pc

Scripts Python:

- **Conversão** de modelo Ultralytics (por exemplo `.pt` → TFLite)
- **Inferência no PC** e **métricas** (COCOeval: mAP, AR, etc.)
- **Avaliação** de JSON exportado pelo app (Flutter / mobile)
- **Gráficos** comparando várias corridas (mAP50, tempo médio de inferência)

Tudo é configurado principalmente no ficheiro **`config.yaml`** (caminhos, dataset, thresholds).

### Estrutura de pastas

| Pasta / ficheiro | Conteúdo |
|------------------|-----------|
| **`src/benchmarks_pc/`** | Código Python (pacote importável `benchmarks_pc`) |
| **`predictions/`** | JSONs exportados pela app e o manifesto `mobile_predictions.example.json` (copie e edite) |
| **`run.py`** (na raiz) | Atalho: adiciona `src/` ao `PYTHONPATH` e chama o CLI |
| **`config.yaml`** | Configuração na raiz de `benchmarks-pc/` |

## Pré-requisitos

- Python 3.10–3.13
- [Poetry](https://python-poetry.org/docs/#installation) 2.x (recomendado) **ou** `pip` + `requirements.txt`

## Instalação com Poetry

Na raiz desta pasta (`benchmarks-pc/`):

```bash
cd benchmarks-pc
poetry install
```

Isto cria um ambiente virtual isolado e instala as dependências do `pyproject.toml`.

### Ativar o ambiente (opcional)

```bash
poetry shell
```

Depois pode usar **`python run.py ...`** na raiz de `benchmarks-pc/` (atalho) ou **`poetry run bench-pc ...`** (comando instalado pelo pacote). Sem `poetry shell`, use `poetry run python run.py ...` ou `poetry run bench-pc ...`.

## Configuração

### eval-mobile e plots: ficheiro de predições vs dataset de GT

O **JSON exportado pela app** é só a lista de **predições** (qual ficheiro ou label avaliar). **Não** define se as métricas são contra COCO ou contra um dataset YOLO.

O **ground truth** — o dataset com que essas predições são comparadas — vem de **`mobile_eval.gt_kind`** e dos caminhos em `paths`:

| `gt_kind` | Onde está o GT | Caminhos usados |
|-----------|----------------|-----------------|
| **`coco`** | Anotações em formato COCO (JSON de instâncias) | `paths.mobile_gt_coco_json`. Nos **plots**, se `plots.gt_json_for_mobile_plots` estiver definido, esse JSON substitui o de `paths` só para os gráficos. |
| **`yolo`** | Imagens + labels `.txt` YOLO (ex.: EPI) | `paths.mobile_gt_yolo_images`, `paths.mobile_gt_yolo_labels`, e nomes de classes via `paths.dataset_yaml` ou `mobile_eval.yolo_class_names`. |

As imagens nas predições têm de corresponder ao mesmo conjunto que o GT (nomes de ficheiro alinhados). Se **omitires** `gt_kind`, o código assume **`coco`**.

1. Edite **`config.yaml`** e preencha pelo menos:
   - **`flutter_assets.models_json`** — lista canónica de modelos do app (por defeito aponta para `apk-yolo-flutter/example/android/app/src/main/assets/models.json`). Os ficheiros **`.tflite`** referenciados no JSON devem existir em **`flutter_assets.models_dir`** (ou, se for `null`, na mesma pasta do `models.json`).
   - **`paths.model_for_inference`** — opcional, fallback quando não há modelo específico por comando; pode ser `null` (usa o `models.json`) ou caminho / nome do `.tflite` em `models_dir`.
   - **`paths.model_for_infer_coco`** — usado só em **`infer-coco`**. Pode ser caminho completo, nome do `.tflite` em `models_dir`, ou o **mesmo texto que o campo `"label"`** em `models.json` (ex.: `yolo11n fp16`). Se vazio, cai no fallback de `model_for_inference` / manifest.
   - **`paths.model_for_infer_epi`** — usado só em **`infer-yolo-gt`**, com as mesmas regras (label do `models.json`, etc.). Sinónimos: `model_for_infer_yolo_gt`, `model_for_yolo_gt`.
   - **`paths.mobile_pred_json`** — para **`eval-mobile`**: caminho para um `.json`, ou o **label** de uma entrada no manifesto `paths.mobile_predictions_json` (se não definir, usa `predictions/mobile_predictions.example.json`).
   - `paths.model_to_convert` — apenas para **`convert`**: caminho do `.pt` a exportar (não vem do `models.json`, que só lista TFLite em runtime).
   - `paths.val_images_dir`, `paths.coco_instances_json` — benchmark com GT COCO
   - Para labels YOLO: `paths.yolo_val_images`, `paths.yolo_val_labels`, `paths.dataset_yaml` (ou nomes em `mobile_eval.yolo_class_names`)
   - Para JSON mobile: `paths.mobile_pred_json` (predições) + **`mobile_eval.gt_kind`** e os paths de GT (`mobile_gt_coco_json` ou `mobile_gt_yolo_*`); ver a subsecção **eval-mobile e plots: ficheiro de predições vs dataset de GT** acima.
   - Para gráficos: `paths.graphs_output_dir` e **ou** `paths.mobile_pred_files` + `model_display_names` (caminhos ou labels resolvidos pelo manifesto), **ou** um manifesto em `paths.mobile_predictions_json` (ver `predictions/mobile_predictions.example.json`). Se `mobile_predictions_json` estiver definido, substitui `mobile_pred_files` por **todas** as runs do ficheiro. Para escolher só algumas, usa **`plots.mobile_pred_labels`**: lista dos mesmos textos que o `"label"` de cada run no manifesto (a ordem da lista é a ordem no gráfico). Opcionalmente `flutter_assets.use_manifest_labels_for_plots: true` preenche nomes a partir do `models.json` quando `model_display_names` está vazio.

3. (Opcional) Usar outro ficheiro de config:

   ```bash
   export BENCHMARKS_PC_CONFIG=/caminho/absoluto/meu_config.yaml
   ```

## Como correr tudo

Todos os comandos devem ser executados **dentro de `benchmarks-pc/`** (ou com `poetry run` a partir daí).

| Comando | O que faz |
|--------|------------|
| `poetry run python run.py convert` | Exporta o modelo (ex.: TFLite) conforme `convert` no YAML |
| `poetry run python run.py infer-coco` | Inferência no conjunto de imagens + avaliação contra annotations COCO JSON |
| `poetry run python run.py infer-yolo-gt` | Mesmo fluxo com ground truth em ficheiros `.txt` YOLO |
| `poetry run python run.py eval-mobile` | Métricas de **um** JSON de predições do app vs GT |
| `poetry run python run.py plots` | Avalia **vários** JSONs mobile e gera gráficos + `comparison_summary.json` |
| `poetry run python run.py list-models` | Lista entradas do `models.json` e se o `.tflite` existe na pasta esperada |
| `poetry run bench-pc …` | Equivalente a `poetry run python run.py …` (script do pacote) |

Com ficheiro de config explícito:

```bash
poetry run python run.py --config /caminho/config.yaml infer-coco
```

### Exemplo de sequência típica

```bash
poetry install
cp config.example.yaml config.yaml
# editar config.yaml com os teus caminhos

poetry run python run.py convert
poetry run python run.py infer-coco
poetry run python run.py eval-mobile
poetry run python run.py plots
```

Saídas por defeito ficam em `paths.output_dir` (ex.: `./benchmark_outputs`), salvo indicação contrária no YAML.

### Modo COCO 80 classes (YOLO pré-treinado no COCO)

No `config.yaml`, em `inference`:

```yaml
class_mapping_mode: coco80
```

Isto usa o mapeamento fixo das 80 classes YOLO para os `category_id` do COCO (como no notebook).

## Instalação alternativa (sem Poetry)

```bash
cd benchmarks-pc
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python run.py --help
# ou: PYTHONPATH=src python -m benchmarks_pc.run --help
```

## Dependências principais

- **ultralytics** — modelo e export
- **pycocotools** — COCOeval
- **matplotlib** — gráficos de comparação
- **PyYAML**, **Pillow**, **tqdm**, **numpy**

## Notas

- A primeira inferência pode demorar (download de pesos, compilação CUDA/CPU, etc.).
- Para datasets grandes (ex.: COCO val completo), reserve espaço em disco e RAM adequados.
- Se usar **GPU**, instale a stack PyTorch compatível com o teu sistema (a Ultralytics puxa dependências; em Linux com CUDA segue a [documentação PyTorch](https://pytorch.org/get-started/locally/)).

## Atualizar dependências (Poetry)

```bash
cd benchmarks-pc
poetry update
```
