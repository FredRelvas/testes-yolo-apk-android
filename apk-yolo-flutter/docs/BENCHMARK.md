# Benchmark de Modelos YOLO

Documento tecnico explicando como funciona o benchmark embarcado no app POC EPI.

---

## O que o benchmark faz

Executa cada modelo `.tflite` bundlado no app por um periodo configuravel
(15s / 30s / 60s) e coleta metricas de **performance**, **energia** e **memoria**.
O resultado e uma tabela comparativa + JSON exportavel.

**O que ele NAO mede:** acuracia / qualidade das deteccoes. Para isso use
a tela "Inferencia em armazenamento" (`StorageInferenceScreen`), que roda
em um dataset real com ground truth.

---

## Como funciona

### 1. Imagem de teste

A inferencia eh feita sobre uma **imagem PNG 640x640 bundlada como asset**
em `example/assets/benchmark_test_image.png`. O app carrega esse arquivo
uma unica vez via `rootBundle.load(...)` no inicio do benchmark e reutiliza
os mesmos bytes para todos os modelos e todas as inferencias.

A imagem real testada eh esta:

![Imagem usada no benchmark](benchmark_test_image.png)

> Este PNG no diretorio `docs/` eh **identico bit-a-bit** ao bundlado no app.

**Composicao da imagem:**

| Elemento | Cor (hex) | Posicao | Tamanho |
|----------|-----------|---------|---------|
| Fundo | `#808080` cinza medio | (0,0)-(640,640) | 640x640 |
| Quadrado escuro | `#404040` | (100,100)-(250,250) | 150x150 |
| Circulo claro | `#CCCCCC` | centro (320,320) | raio 150 |
| Quadrado branco | `#E8E8E8` | (420,420)-(540,540) | 120x120 |

**Garantia de determinismo:** como a imagem eh um asset estatico embarcado
no APK, **todas** as execucoes do benchmark em **qualquer dispositivo** ou
versao do Flutter usarao exatamente os mesmos pixels como input. Isso
elimina qualquer variacao de anti-aliasing entre versoes do engine Skia
que poderia ocorrer se a imagem fosse gerada em runtime.

**Por que nao usa camera?**

O custo computacional do YOLO **independe do conteudo da imagem**: o input
passa pelas mesmas convolucoes na rede neural, e detectar 0 ou 100 objetos
faz diferenca marginal so no post-processing (NMS). Para o objetivo de
comparar **modelos entre si**, a imagem sintetica garante:

- Mesmo input em todas as execucoes -> resultados comparaveis
- Sem permissao de camera
- Sem dependencia do que esta em frente ao dispositivo
- A camera consome ~300-500 mA por si so, que poluiria a medicao de energia
  do modelo se estivesse ativa

**Codigo que carrega a imagem** -- ver `_loadTestBitmap()` em
`example/lib/presentation/screens/benchmark_screen.dart`.

### 2. Sequencia de cada execucao

Para cada (modelo + acelerador), o benchmark executa **4 etapas**:

```
[1. Loading]    Carrega o modelo via YOLO(...) + loadModel()
                e zera o contador de mAh.

[2. Warmup ]    Roda inferencias em loop por 5 segundos.
   5s           Necessario para compilar shaders GPU, aquecer
                JIT do interpretador TFLite e preencher caches.
                Resultados desta fase sao DESCARTADOS.

[3. Estabilizacao] Aguarda 5 segundos SEM inferencias.
   5s              Permite que caches assentem e memoria estabilize.

[4. Coleta]     Roda inferencias em loop por N segundos
   N segundos   (15/30/60). Esta eh a fase MEDIDA:
                - Stopwatch ao redor de cada predict() -> infTimes
                - Listener no SystemMetricsService -> mA + RAM + mAh
                
[5. Dispose]    Chama yolo.dispose() que libera Interpreter TFLite
                + GpuDelegate na camada nativa. Sem isso, recursos
                acumulariam (memory leak).
```

**Tempo total por execucao:** `10s + N` (overhead fixo de aquecimento)

**Tempo total do benchmark:** `numero_de_execucoes * (10 + N)` segundos

### 3. Modos de acelerador

| Modo | Comportamento | N execucoes |
|------|---------------|-------------|
| **GPU** | Todos os modelos com `useGpu: true` | 1 por modelo |
| **CPU** | Todos os modelos com `useGpu: false` | 1 por modelo |
| **Ambos** | Cada modelo roda 2 vezes: GPU + CPU | 2 por modelo |

No modo Ambos, a coluna "GPU" do relatorio (Sim/Nao) diferencia as variantes,
permitindo comparar diretamente o ganho de cada modelo com aceleracao.

---

## Metricas coletadas

### Performance

| Metrica | O que significa | Fonte |
|---------|-----------------|-------|
| **Avg ms** | Tempo medio de uma inferencia em milissegundos | `Stopwatch` no predict() |
| **Min ms / Max ms** | Extremos do tempo de inferencia | idem |
| **FPS** | Frames por segundo = `totalInferences / durationSec` | calculado |
| **N inf** | Numero total de inferencias na fase de coleta | contador |

**Interpretacao:** Avg ms eh a metrica mais importante. Menor = mais rapido.
FPS eh inverso de Avg ms, util para entender a taxa sustentada que o modelo
aguenta em camera ao vivo.

### Energia

| Metrica | O que significa | Fonte |
|---------|-----------------|-------|
| **Avg mA** | Corrente media drenada da bateria durante a coleta | `BatteryManager.CURRENT_NOW` (Android) |
| **mAh** | Energia total consumida durante a coleta | Integral de mA ao longo do tempo |

**Importante sobre mA:** essa eh a corrente do **dispositivo inteiro**,
nao isolada ao app. Inclui tela, radios, outros processos. Por isso serve
bem para **comparar cenarios** (modelo A vs B no mesmo dispositivo), mas
nao representa o consumo absoluto so do app.

**Como o mAh eh calculado:**
```
A cada tick de 1 segundo:
  sessionMah += currentMa * (1 / 3600)
```
Ou seja, integra a corrente ao longo do tempo. O benchmark zera o contador
no inicio de cada execucao (`resetSession()`) e captura o valor acumulado
no final da fase de coleta.

**Interpretacao:** menor = mais eficiente energeticamente. Compare modelos
do mesmo tipo (ex: yolo26n fp32 vs fp16) para entender se a quantizacao
vale a pena no seu dispositivo.

### Memoria

| Metrica | O que significa | Fonte |
|---------|-----------------|-------|
| **Avg RAM MB** | Memoria residente media do processo Flutter durante a coleta | `ProcessInfo.currentRss` (dart:io) |

**Importante:** essa eh a memoria **Java/Dart** do processo. A memoria
nativa do TFLite Interpreter + buffers GPU **nao aparece aqui** (e
geralmente bem maior). Mas eh suficiente para detectar memory leaks
entre trocas de modelo: se o valor cresce continuamente, ha algo
escapando do dispose.

---

## Fluxo do usuario

### Fase 1: Setup

Tela inicial mostra:
- Lista de modelos disponiveis (vem de `ModelRegistry.instance.all`)
  com checkbox individual (todos marcados por padrao)
- Selecionador de acelerador: GPU / CPU / Ambos
- Dropdown de duracao por modelo: 15s / 30s / 60s
- Texto com tempo estimado total ("~X min")
- Botao "Iniciar (N execucoes)"

### Fase 2: Running

Durante a execucao mostra:
- Barra de progresso global (modelo N de total)
- Card do modelo atual com nome e modo (GPU/CPU)
- Etapa atual (Carregando / Aquecendo / Estabilizando / Coletando)
- Countdown da etapa atual em segundos
- Metricas ao vivo: FPS atual, inf ms medio, mA instantaneo

Botao "Cancelar" para parar a qualquer momento -- modelo atual eh
disposed corretamente e o relatorio mostra o que foi coletado ate ali.

### Fase 3: Report

`DataTable` com colunas:
- **Modelo** -- label do modelo
- **GPU** -- Sim / Nao
- **Avg ms** -- tempo medio de inferencia
- **FPS** -- frames por segundo equivalente
- **Avg mA** -- corrente media
- **mAh** -- energia total da coleta
- **RAM MB** -- memoria media
- **N inf** -- numero de inferencias

**Destaques visuais:**
- Linha **verde**: modelo com menor `Avg ms` (mais rapido)
- Linha **azul**: modelo com menor `mAh` (mais eficiente energeticamente)
- Linha **verde-azulada**: se um modelo eh ambos

Botoes:
- **Compartilhar** -- exporta JSON via share_plus (WhatsApp, e-mail, etc)
- **Novo teste** -- volta para a fase Setup zerando os resultados

---

## Formato do JSON exportado

```json
{
  "timestamp": "2026-05-12T14:30:00.000",
  "duration_per_model_sec": 30,
  "warmup_sec": 5,
  "stabilization_sec": 5,
  "results": [
    {
      "model_label": "EPI YOLO26m fp32",
      "model_file": "exp_yolo26m_epi_negative_float32.tflite",
      "use_gpu": true,
      "total_inferences": 124,
      "avg_inf_ms": 241.523,
      "min_inf_ms": 198.012,
      "max_inf_ms": 412.876,
      "avg_fps": 4.13,
      "avg_ma": 1487.3,
      "total_mah": 12.394,
      "avg_ram_mb": 287.4,
      "duration_sec": 30.02
    }
  ]
}
```

Cada entrada em `results` corresponde a uma execucao (modelo + acelerador).
No modo "Ambos", aparecem 2 entradas por modelo selecionado.

---

## Premissas e limitacoes

### O que afeta os resultados

- **Estado do dispositivo:** se a CPU/GPU ja estao quentes de outro
  workload, os primeiros modelos podem rodar levemente mais rapido.
  O warmup de 5s mitiga isso, mas nao elimina completamente.
- **Bateria carregando:** durante carga, o `CURRENT_NOW` pode ficar
  negativo ou nao refletir o consumo real do app. **Rode o benchmark
  com o cabo desconectado** para medicoes de energia validas.
- **Apps em background:** sincronizacoes, downloads ou animacoes de
  outros apps competem por CPU/GPU. Feche o que puder antes de rodar.
- **Modo de bateria:** alguns dispositivos limitam clock em modo
  economizador de energia. Desative para medir o pico de performance.

### O que o benchmark NAO testa

- **Acuracia das deteccoes** -- use `StorageInferenceScreen`
- **Pipeline real da camera** -- preprocessing varia com fonte do frame
- **Latencia ponta-a-ponta** -- so o tempo de `predict()`, sem renderizacao
- **Termal throttling de longa duracao** -- benchmarks curtos (15-60s)
  raramente disparam reducao termal. Para testes longos, configure
  uma duracao maior ou repita em sequencia.

---

## Arquivos relevantes

| Arquivo | Funcao |
|---------|--------|
| `example/lib/presentation/screens/benchmark_screen.dart` | Tela principal com as 3 fases |
| `example/lib/models/benchmark_model_result.dart` | Data class do resultado por execucao |
| `example/lib/services/system_metrics_service.dart` | Coletor de mA, RAM e bateria a 1Hz |
| `example/lib/services/model_manager.dart` | Resolucao de paths e carregamento dos modelos |
| `example/lib/services/model_registry.dart` | Lista de modelos disponiveis (do `models.json`) |
| `lib/yolo.dart` | API do plugin (YOLO + loadModel + predict + dispose) |

---

## Como substituir a imagem de teste

A imagem eh um arquivo PNG bundlado em
`example/assets/benchmark_test_image.png`. Para usar outra imagem
no benchmark:

1. Substitua o arquivo mantendo o mesmo nome e caminho
2. Idealmente, mantenha 640x640 (tamanho de input do YOLO) -- se trocar,
   o plugin nativo fara resize, o que pode adicionar overhead que nao
   reflete o custo real do modelo
3. Recompile o app

Para regenerar a imagem padrao via script Python (Pillow):

```python
from PIL import Image, ImageDraw

img = Image.new('RGB', (640, 640), (0x80, 0x80, 0x80))
draw = ImageDraw.Draw(img)
draw.ellipse([(170, 170), (470, 470)], fill=(0xCC, 0xCC, 0xCC))
draw.rectangle([(100, 100), (250, 250)], fill=(0x40, 0x40, 0x40))
draw.rectangle([(420, 420), (540, 540)], fill=(0xE8, 0xE8, 0xE8))
img.save('example/assets/benchmark_test_image.png')
```
