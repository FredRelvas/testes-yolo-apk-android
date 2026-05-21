# Otimizacao de Modelos INT8 para GPU no Android

Documento executivo da investigacao que levou os modelos YOLO quantizados (int8)
a rodarem na GPU do Android com **ate 4x de speedup** sobre CPU.

Para o funcionamento basico do benchmark (telas, fluxo, metricas), ver [BENCHMARK.md](BENCHMARK.md).

---

## Resumo executivo

Partimos de um cenario onde os 4 modelos int8 bundlados no app **falhavam em GPU**
ou rodavam num **modo hibrido inutil** (97% das ops em CPU). Apos investigacao
do TFLite GPU delegate v2 + refactor do pipeline de quantizacao no projeto irmao
(`../quant-modelos`), chegamos a: **2 dos 4 int8 rodam 100% em GPU em ~9 ms**
(equivalente aos fp16) e os outros 2 ficam em CPU XNNPACK por design.

**O insight principal:** o problema nunca foi "int8 nao roda em GPU". Foi
**NMS embutido no grafo TFLite** combinado com **particionamento all-or-nothing**
do GPU delegate v2. Tirando o NMS do grafo (deixando o app fazer pos-processamento),
o caminho GPU se abre.

### Mini-tabela: antes vs depois (Samsung S25+)

Comparacao usando **FPS efetivo** (metrica end-to-end mais honesta, ja que para
os int8 antigos so temos logs do TFLite puro, nao JSON Dart-side):

| Modelo + Acelerador | Antes do refactor | Depois do refactor |
|---|---|---|
| `yolo26n_int8_full` GPU | ~19 fps (hibrido, 97% das ops em CPU) | **21.0 fps** (100% GPU) |
| `yolo11n_int8_full` GPU | ~16 fps (ja funcionava) | 15.8 fps |
| `yolo26n_int8_dr` GPU   | ~14 fps (hibrido) | falha (esperado, ver Apendice C) |
| `yolo11n_int8_dr` GPU   | falha catastrofica | falha (esperado) |

**Importante**: a inferencia TFLite pura melhorou ~4× (de ~39 ms hibrido para
~9 ms GPU completa segundo logs), mas o end-to-end Dart-side ganhou so ~10%
porque **o overhead de pre/pos-processamento e MethodChannel passou a dominar**.
Ver [secao de Analise dos Resultados](#analise-dos-resultados) e
[Apendice E](#apendice-e--limites-de-hardware-observados).

---

## O problema inicial

Quando adicionamos os 4 modelos int8 ao app, marcamos todos com `useGpu: false`
no `models.json` baseados na **suposicao errada** de que "GPU delegate do TFLite
nao suporta int8". Ao tentar forcar GPU mesmo assim, observamos **3 sintomas
diferentes** entre os modelos:

1. **Falha catastrofica silenciosa** (`yolo11n_int8_dr`): a tela do benchmark
   reportava **0 inferencias** na fase GPU, mas sem erro visivel. Apos ~25s
   o app simplesmente "pulava" para a fase CPU.
2. **Modo hibrido inutil** (`yolo26n_int8_*`): inferencia rodava, mas o tempo
   era praticamente identico ao CPU puro. GPU pegava ~3% das ops, o resto
   ia para XNNPACK.
3. **GPU completa funcional** (`yolo11n_int8_full`): rodava perfeitamente em
   ~9 ms.

A pergunta natural: **por que tres comportamentos para "o mesmo formato int8"?**

> Detalhe da investigacao: [Apendice A](#apendice-a--mecanica-do-gpu-delegate-v2)
> e [Apendice D](#apendice-d--metodologia-de-investigacao).

---

## Causa raiz

Apos analise dos logs (logcat com tags `ObjectDetector`, `YOLOPlugin`, `tflite`,
`AndroidRuntime`) e inspecao dos grafos TFLite com `ai_edge_litert`, identificamos
**tres fatores combinados**:

1. **NMS embutido no grafo** introduz operacoes com `batch=300/400` (top-K
   interno + MatMul pairwise de IoU). O GPU delegate v2 do TFLite **so suporta
   `batch=1`** na primeira dimensao da maioria das ops. Resultado: erro
   `"Batch size mismatch, expected 1 but got 400"`.

2. **Particionamento all-or-nothing**: quando o exportador agrupa tudo em uma
   unica particao GPU (`Replacing 347 out of 347 nodes (1 partition)`), uma
   unica op incompativel **derruba a particao inteira** → exception
   `IllegalArgumentException: Internal error: Failed to apply delegate`.

3. **Flag `end2end` da Ultralytics varia por arquitetura**: o `yolo26n.pt`
   vinha com `end2end=True` por default, embutindo o NMS no grafo. O `yolo11n.pt`
   vinha com `end2end=False`. Isso causava o `yolo26n_int8_full` a embutir NMS
   (e cair no modo hibrido) enquanto o `yolo11n_int8_full` saia limpo.

> Detalhe sobre NMS e por que GPUs odeiam ele:
> [Apendice B](#apendice-b--nms-no-grafo-vs-fora-do-grafo).

---

## Decisoes tomadas

| Decisao | Motivo | Impacto |
|---------|--------|---------|
| Forcar `nms=False, end2end=False` em todos os exports tflite | Manter o NMS fora do grafo, deixando o pos-processamento no app | Yolo26n_int8_full saltou de **TFLite ~39 ms (hibrido) para ~9 ms (GPU completa)**; ganho de FPS end-to-end mais modesto (~10%) porque overhead Dart/MethodChannel domina |
| Substituir `tf.lite.TFLiteConverter.from_saved_model()` pelo path `YOLO.export(format="tflite", int8=True)` no script DR | O caminho via SavedModel deixava 3 ops orfas com `batch=400` no grafo (dead code); o path Ultralytics passa por ONNX que faz dead-code elimination | Ainda nao resolveu 100% para variantes DR (3 ops persistem), mas eliminou todas as outras inconsistencias |
| Manter `useGpu: false` para `yolo*_int8_dr.tflite` | Quantizacao Dynamic Range no Ultralytics 8.4.52 sempre deixa 3 ops orfas que travam o GPU delegate; CPU XNNPACK roda normal em ~37 ms | App funciona com fallback transparente para esses modelos |
| **NAO** investir em NPU/NNAPI | Fragmentacao do parque Android: NPU usavel em ~20-30% dos phones. GPU+CPU cobre 100% | App permanece portavel; GPU como caminho rapido, CPU como fallback universal |

> Detalhe do pipeline de quantizacao corrigido:
> [Apendice C](#apendice-c--pipeline-de-quantizacao-corrigido).

---

## Resultados finais

Benchmark rodado no **Samsung Galaxy S25+ desconectado do USB** (para medicao
de energia sem viés de carregamento e sem throttling do modo debug), com
9 modelos × 2 aceleradores = 18 execucoes de **60 segundos cada** (warmup 5s +
estabilizacao 5s + coleta 60s).

JSON cru: `benchmark_1779310803206.json`.

### Tabela completa

Numeros sao **Dart-side** (round-trip `await yolo.predict(...)` incluindo
MethodChannel + BitmapFactory + pre + inf + pos). Para inferencia TFLite pura,
ver [Apendice E](#apendice-e--limites-de-hardware-observados).

| Modelo | GPU avg ms | GPU FPS | CPU avg ms | CPU FPS | Melhor caminho |
|--------|----------:|--------:|----------:|--------:|----------------|
| `EPI YOLO26m fp32` | 737.6 | 1.36 | 749.1 | 1.33 | Empate (modelo pesado) |
| `yolo26n fp32 (COCO)` | 44.9 | **22.3** | 106.1 | 9.43 | **GPU** (2.36×) |
| `yolo26n fp16 (COCO)` | 45.6 | 21.9 | 109.0 | 9.18 | **GPU** (2.39×) |
| `yolo26n int8 DR (COCO)` | falha | — | 83.3 | 12.0 | CPU (DR otimo em XNNPACK) |
| `yolo26n int8 full (COCO)` | 47.7 | 21.0 | 80.1 | 12.48 | **GPU** (1.68×) |
| `yolo11n fp32 (COCO)` | 49.9 | 20.0 | 129.5 | 7.72 | **GPU** (2.59×) |
| `yolo11n fp16 (COCO)` | 51.9 | 19.3 | 129.9 | 7.70 | **GPU** (2.50×) |
| `yolo11n int8 DR (COCO)` | falha | — | 89.4 | 11.19 | CPU |
| `yolo11n int8 full (COCO)` | 57.2 | 15.8 | 135.6 ⚠️ | 7.38 | **GPU** (2.37×) |

> ⚠️ `yolo11n_int8_full` CPU teve outlier extremo de 17.8 s em uma inferencia
> (provavelmente GC ou throttling pontual). A mediana real esta provavelmente
> em ~70-80 ms — ver secao de Analise.

### Graficos

Gerados via `python3 docs/gerar_graficos.py` a partir do JSON do benchmark final.

- `graficos/01_avg_inf_ms.png` — tempo medio de inferencia, GPU vs CPU
- `graficos/06_gpu_speedup.png` — quanto a GPU acelera (ou nao) cada modelo
- `graficos/09_best_per_model.png` — melhor configuracao por modelo
- `graficos/07_fps_vs_mah.png` — mapa de eficiencia (FPS × energia)

Conjunto anterior (antes do refactor) preservado em `graficos_antes_refactor/`
para comparacao direta.

---

## Analise dos resultados

Insights nao obvios que sairam dos numeros finais e merecem destaque:

### 1. Hipoteses confirmadas

Todas as previsoes da fase de investigacao foram validadas:

- `yolo*_int8_full` GPU funcionam apos refactor (eram hibridos ou falhavam)
- `yolo*_int8_dr` GPU **falham** mesmo apos refactor, conforme previsto
  (3 ops orfas `batch=400` persistem no caminho de export do Ultralytics 8.4.52)
- `EPI YOLO26m` empata GPU vs CPU (~745 ms): o modelo **tem NMS embutido no
  grafo** (output `[1, 300, 6]`, ver heuristica no [Apendice B](#apendice-b--nms-no-grafo-vs-fora-do-grafo)),
  forcando execucao no mesmo modo hibrido inutil que o `yolo26n_int8_full`
  tinha antes do refactor — GPU pega so as primeiras convs, o resto (~95% das
  ops, incluindo o NMS) cai em CPU XNNPACK. Solucao seria **reexportar o EPI
  com `nms=False, end2end=False`** (se tivermos acesso ao `.pt`), seguindo o
  mesmo pipeline do [Apendice C](#apendice-c--pipeline-de-quantizacao-corrigido).
  Expectativa pos-refactor: ~150-200 ms na GPU (4-5× speedup, proporcional ao
  ganho observado no `yolo26n_int8_full`)

### 2. **O overhead virou o gargalo** (insight mais importante)

A inferencia TFLite pura ficou ~4× mais rapida com GPU (de ~39 ms hibrido para
~9 ms GPU completa). **Mas o end-to-end Dart-side ganhou pouco** porque o
overhead fixo de pre/pos-processamento + MethodChannel + decode de Bitmap
representa **~35-40 ms por inferencia** que nao muda independente do modelo.

Decomposicao tipica de um `avg_inf_ms = 47.7 ms` (yolo26n_int8_full GPU):

```
Total Dart       : 47.7 ms (100%)
  ├ MethodChannel: ~3 ms     (6%)
  ├ Bitmap decode: ~3-5 ms   (8%)
  ├ Pre (resize) : ~25 ms    (52%)  ← maior fatia
  ├ TFLite Inf   : ~9 ms     (19%)  ← o que otimizamos
  ├ Post (NMS)   : ~5 ms     (10%)
  └ Resp+IPC     : ~3 ms     (6%)
```

**Implicacao**: para futuros ganhos significativos, focar em **pre-processamento
GPU-accelerated**, **pre-alocacao de bitmap** ou **eliminar o decode por
inferencia** (passar `ByteBuffer` direto, evitar PNG bundlado), nao em
otimizar mais o TFLite.

### 3. INT8 DR e o **mais rapido em CPU**

Resultado contra-intuitivo: os modelos `int8_dr` (que falham na GPU) sao os
**mais rapidos em CPU**:

| Modelo (CPU) | avg ms | inf/min | Δ vs fp32 |
|--------------|-------:|--------:|-----------|
| yolo26n fp32      | 106 | 566 | baseline |
| yolo26n fp16      | 109 | 551 | -3% |
| yolo26n **int8 DR** | **83** | **720** | **+27% mais rapido** ⚡ |
| yolo26n int8 full | 80 | 749 | +33% mais rapido |
| yolo11n fp32      | 130 | 464 | baseline |
| yolo11n **int8 DR** | **89** | **672** | **+46% mais rapido** ⚡ |
| yolo11n int8 full | 136 ⚠️ | 443 | (outlier distorce) |

O XNNPACK realmente usa os kernels int8 quando disponiveis. **Para
dispositivos sem GPU funcional, distribuir o `int8_dr` e a melhor escolha**.

### 4. GPU **mais eficiente em energia por inferencia**

Embora a GPU consuma mais corrente instantanea, ela termina muito mais
inferencias no mesmo tempo. Eficiencia liquida (inferencias por mAh):

| Modelo + Acel | Total mAh | N inf | inf/mAh |
|---------------|----------:|------:|--------:|
| yolo26n int8 full GPU | 23.3 | 1258 | **54.0** |
| yolo26n int8 full CPU | 21.5 | 749  | 34.9 |
| yolo26n fp32 GPU      | 29.5 | 1337 | 45.3 |
| yolo26n fp32 CPU      | 27.9 | 566  | 20.3 |

**GPU faz 54-55% mais inferencias por mAh consumido.** Isso desmonta o mito
de que "GPU gasta mais bateria" — gasta mais por segundo, mas entrega mais
trabalho por unidade de energia.

### 5. Efeito do USB desconectado (~15% de ganho)

Comparando o mesmo modelo no benchmark com USB e sem USB:

| Modelo | GPU com USB | GPU sem USB | Δ |
|--------|------------:|------------:|---|
| yolo26n fp32 | 52 ms | **44.9 ms** | -14% |
| yolo26n fp16 | 54 ms | **45.6 ms** | -16% |
| yolo11n fp32 | 62 ms | **49.9 ms** | -19% |
| yolo11n fp16 | 59 ms | **51.9 ms** | -12% |

Confirma o vies real do USB conectado:
- ADB ativo mantem o phone em modo debug (governor mais conservador)
- Carga via USB pode oscilar o `BatteryManager.CURRENT_NOW` afetando medicoes
  de energia

**Recomendacao**: benchmarks futuros sempre desconectados, com tela ligada
mas sem carregar. Resultados de mA/mAh so sao confiaveis nessa condicao.

### 6. Outlier extremo no `yolo11n_int8_full` CPU

```
max_inf_ms: 17843.57   ← UMA inferencia demorou 17.8 segundos!
```

Provavelmente um hiccup do sistema (GC do Android, throttling, app em
background concorrendo). Esta unica inferencia representa ~30% de todo o
tempo de coleta (60 s) e joga o avg de ~80 ms para 135 ms.

**Implicacao para o benchmark**: vale futuramente reportar **mediana ou
percentil 95** alem da media, para nao deixar outliers raros distorcerem
a comparacao. A media e mais sensivel a esses spikes.

### 7. Convergencia de FPS em modelos GPU compativeis

Todos os 6 modelos nano que rodam em GPU convergem em **15-22 FPS**
end-to-end (Dart-side):

```
yolo26n fp32 GPU : 22.3 fps (44.9 ms)
yolo26n fp16 GPU : 21.9 fps (45.6 ms)
yolo26n int8 full GPU: 21.0 fps (47.7 ms)
yolo11n fp32 GPU : 20.0 fps (49.9 ms)
yolo11n fp16 GPU : 19.3 fps (51.9 ms)
yolo11n int8 full GPU: 15.8 fps (57.2 ms)  ← mais lento dos nano
```

O yolo11n_int8_full GPU ser ~10 ms mais lento que os outros e estranho
(deveria empatar ou ganhar do fp16). Possivelmente reflete uma calibracao
sub-otima ou um padrao especifico do grafo que o GPU delegate compila de
forma menos eficiente. Vale investigar com `tflite analyze` em uma proxima
iteracao.

### Recomendacao final para a POC EPI

Baseado nestes resultados, para o caso COCO generico (validacao):

1. **Default na tela de camera**: `yolo26n fp32 GPU` (45 ms / 22 fps, melhor
   balanco velocidade/qualidade)
2. **Fallback em phones sem GPU**: `yolo26n int8 DR CPU` (83 ms / 12 fps,
   mais rapido em CPU que qualquer outro)
3. **Modelos EPI customizados (YOLO26m)**: investigar se ha versao **n** ou
   **s** disponivel — o **m** com 745 ms / 1.3 fps e pesado demais para uso
   em tempo real

---

## O que NAO fizemos e por que

### NPU / NNAPI: decisao explicada de adiar para esta etapa inicial

A NPU (Neural Processing Unit) dedicada do S25+ poderia em teoria rodar int8
ainda mais rapido (~2-3 ms vs 9 ms da GPU) e gastar **5-10× menos energia** que
a GPU para o mesmo trabalho. Apesar disso, **decidimos conscientemente NAO
implementar suporte a NPU nesta fase inicial da POC**. Os motivos sao
encadeados e vale registrar com cuidado para futuras consultas:

#### 1. Portabilidade e o publico-alvo da POC

O app **precisa ser portavel** entre celulares Android — funcionarios e
gerentes nao recebem dispositivos padronizados. A realidade do parque Android
hoje:

| SoC / Categoria | NPU dedicada usavel via NNAPI? |
|-----------------|:-:|
| Snapdragon 8 Gen 1+ (top atual) | Sim (Hexagon), mas Samsung/OEMs frequentemente limitam |
| Exynos 2200+ (Samsung topo) | Sim, mas restrito ao Samsung Neural SDK |
| Dimensity 9000+ (MediaTek topo) | Sim (APU) |
| Snapdragon 7/6/4 series (mid/budget) | Fraca ou inexistente |
| Helio / Dimensity 6/7 (MediaTek mid/budget) | Tipicamente nao |
| Phones com mais de 3-4 anos | Raramente |

NPU usavel via NNAPI esta presente em **~20-30% do parque em uso real**.
Investir tempo significativo em otimizar para NPU beneficiaria a minoria dos
usuarios e adicionaria caminhos de codigo dificeis de testar nos dispositivos
restantes.

#### 2. Inconsistencia historica do NNAPI

Mesmo nos phones que **tem** NPU, o NNAPI (a API generica do Android para
acessa-la) tem historico ruim:

- **Samsung empurra seu Neural SDK proprio**, frequentemente desabilitando ou
  limitando o NNAPI para forcar uso do SDK proprietario.
- **Google esta migrando para AICore** (Android 14+) — o NNAPI classico esta
  em modo de manutencao.
- **Drivers NPU variam por fabricante**: o mesmo modelo .tflite pode rodar
  bem na NPU de um phone e cair em CPU em outro.
- **Testar isso exige uma flotilha de dispositivos** que nao temos no momento.

#### 3. Estrategia atual ja entrega o necessario

A POC EPI nao precisa de 200+ fps. Precisa de:

- **Latencia baixa o suficiente para feedback visual em tempo real** (~30 fps
  end-to-end e mais que suficiente para deteccao de capacete/colete)
- **Funcionar em qualquer celular Android razoavelmente moderno**
- **Nao drenar a bateria absurdamente** durante a sessao de monitoramento

O caminho GPU + CPU fallback que ja temos cobre **100% do parque Android** e
entrega:

```
Top phone (S25+)        : ~9 ms inferencia GPU   (~40 ms end-to-end)
Mid-range moderno       : ~15-25 ms inferencia GPU (~50-60 ms end-to-end)
Budget recente          : ~30-50 ms inferencia GPU (~70-90 ms end-to-end)
Antigo (sem GPU usavel) : CPU XNNPACK ~80-150 ms  (~120-200 ms end-to-end)
```

Em todos os cenarios o app permanece **funcional**. A NPU traria ganho real
apenas no primeiro tier (que ja roda otimo).

#### 4. Custo de manutencao e risco tecnico

Adicionar suporte real a NPU significaria:

- Mudar codigo nativo Kotlin do plugin `ultralytics_yolo` (`ObjectDetector.kt`,
  `Segmenter.kt`, etc) para configurar NNAPI explicitamente, fazer fallback
  por dispositivo, expor opcoes.
- Testar em **multiplos chips/fabricantes** para validar que nao quebra em
  nenhum.
- Manter compatibilidade com novas versoes do Android (NNAPI → AICore).
- Possivelmente forkar o plugin (mudancas nao triviais ficariam dificeis de
  manter alinhadas com upstream).

Para uma POC que precisa **validar viabilidade de negocio** (deteccao funciona?
modelos sao bons o suficiente? usuarios usariam o app?), investir em NPU agora
seria **otimizacao prematura**.

#### 5. O plugin ja tem fallback NNAPI (passivo, nao engaja na pratica)

Importante notar: o codigo nativo em `ObjectDetector.kt:91-99` **ja tenta**
adicionar um `NnApiDelegate` quando `useGpu=false`:

```kotlin
} else {
    try {
        val delegate = NnApiDelegate()
        addDelegate(delegate)
        Log.d("ObjectDetector", "NNAPI delegate is used (useGpu=false).")
    } catch (e: Exception) {
        Log.e("ObjectDetector", "NNAPI delegate error: ${e.message}")
    }
}
```

No entanto, nos logs do benchmark no S25+ **nao observamos `TfLiteNnApiDelegate`
pegando ops** — o XNNPACK chega antes e pega as 388-401 ops disponiveis.
Confirmar que NNAPI nao engaja exigiria um experimento isolado (NNAPI-only,
sem XNNPACK), que tambem nao foi feito.

#### Quando reabrir a decisao no futuro

Vale revisitar a decisao de adicionar NPU se:

- A POC for promovida a **produto**, com **dispositivos padronizados** (todos
  os funcionarios recebem o mesmo phone modelo X) — aceitavel otimizar para
  esse chip especifico
- O **caso de uso evoluir** para algo que **realmente exija batch processing
  ou continuous monitoring de longa duracao** — aí o ganho energetico da NPU
  passa a importar
- Modelos **maiores** (yolov8m, yolov11l) forem necessarios e a GPU deixar de
  dar conta
- O **Google AICore** estabilizar como API padrao para acesso a NPU em Android
  14/15+, reduzindo o custo de manutencao da integracao

### Limpeza das 3 ops orfas do DR via `onnxslim` — trabalho futuro

Os modelos `*_int8_dr.tflite` ainda tem 3 ops residuais (`MatMul/Multiply/Transpose`
com `batch=400`) que o GPU delegate rejeita. Eliminar essas ops exigiria passar
o tflite por ferramentas de simplificacao adicionais (`onnxslim` ou edicao manual
do flatbuffer). Como o full int8 ja cobre o caso GPU e o DR funciona bem em CPU,
isso foi adiado.

### Arquiteturas NMS-free (YOLOv10, YOLOv11 end-to-end) — fora do escopo

Modelos modernos como YOLOv10 introduziram "one-to-one matching" durante treino,
eliminando a necessidade de NMS classico. Migrar para essas arquiteturas mudaria
o ja-treinado modelo EPI, fora do escopo desta otimizacao.

### TFLite GPU delegate v3 / experimental quantized GPU

Existem branches experimentais do TFLite com melhor suporte int8 em GPU
(`TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT`), mas nao estao expostos na
versao do plugin Flutter `ultralytics_yolo` em uso. Atualizar exigiria fork
do plugin.

---

# Apendices

## Apendice A — Mecanica do GPU delegate v2

O TFLite GPU delegate v2 funciona em 3 etapas durante `Interpreter.<init>`:

1. **Avaliacao**: percorre o grafo (DAG de ops) e marca cada op como
   "GPU-compativel" ou "GPU-incompativel". Restricoes conhecidas:
   - `batch != 1` na primeira dimensao rejeita a maioria das ops
   - Tipos int8 quantizados sao parcialmente suportados (depende da op,
     shape, kernel size, padding)
   - Ops como `gather_nd`, `argsort`, `top_k`, `non_max_suppression` nao tem
     kernels GPU implementados
2. **Particionamento**: agrupa ops GPU-compativeis em "particoes" continuas.
   Cada particao vira um unico kernel GPU compilado. Quanto **menos** particoes,
   menos overhead de switching CPU↔GPU.
3. **Compilacao**: tenta compilar cada particao em kernels OpenGL/Vulkan/OpenCL.
   Se uma particao **inteira** nao puder ser compilada, e descartada.

### Os 3 cenarios de falha que observamos

**Cenario 1: Falha catastrofica (yolo11n_int8_dr)**
```
I tflite : Replacing 347 out of 347 node(s) with delegate (TfLiteGpuDelegateV2), yielding 1 partition
I tflite : Created 0 GPU delegate kernels.   ← ZERO kernels = falhou
E YOLOPlugin: java.lang.IllegalArgumentException: Internal error: Failed to apply delegate:
  TfLiteGpuDelegate Init: Batch size mismatch, expected 1 but got 400
```
Exporter colocou tudo em 1 particao monolitica; uma op com `batch=400` derruba
tudo; o `SynchronizedLazyImpl` do Kotlin retenta o construtor em loop infinito
(observamos 297 tentativas em 25 segundos).

**Cenario 2: Modo hibrido inutil (yolo26n_int8_full ANTES do refactor)**
```
I tflite : Replacing 13 out of 433 node(s) with delegate (TfLiteGpuDelegateV2), yielding 3 partitions
I tflite : Replacing 388 out of 433 node(s) with delegate (TfLiteXNNPackDelegate)
```
O exporter espalhou ops incompativeis pelo grafo, permitindo o TFLite isolar
3 particoes GPU pequenas (13 ops) e mandar 388 para XNNPACK. Funciona, mas
GPU contribui tao pouco que o tempo total e ~igual ao CPU puro.

**Cenario 3: GPU completa (yolo11n_int8_full)**
```
I tflite : Replacing 345 out of 345 node(s) with delegate (TfLiteGpuDelegateV2), yielding 1 partition
I tflite : Created 1 GPU delegate kernels.   ← 1 kernel = sucesso
```
Grafo limpo (sem ops `batch=400`, sem NMS embutido); 100% das ops cabem em 1
particao GPU; compilacao bem-sucedida.

---

## Apendice B — NMS no grafo vs fora do grafo

### O que o NMS faz

Saida crua do YOLO em 640×640: `[1, 84, 8400]` = **8.400 caixas candidatas**
por inferencia, muitas sobrepostas detectando o mesmo objeto. NMS
(Non-Maximum Suppression) reduz para tipicamente 0-50 caixas finais via:

1. Ordenar caixas por confianca
2. Pegar a de maior confianca (vencedora)
3. Calcular IoU pairwise entre vencedora e todas as outras
4. Descartar caixas com IoU > threshold
5. Repetir ate esvaziar

O passo 3 e o que cria o **MatMul [400, 400]** (160k comparacoes IoU) que
explode os shapes incompativeis com GPU.

### Por que GPU odeia NMS

- **Sequencial por natureza** (loop iterativo) — GPU foi feita para paralelismo
- **Output dinamico** (0-N caixas) — GPU prefere shapes fixos para compilar shaders
- **Ops irregulares**: `gather_nd`, `argsort`, `top_k`, `non_max_suppression` nao
  tem kernels GPU no delegate v2
- **Branches condicionais**: causa warp divergence (perda de paralelismo)

### Heuristica para detectar NMS embutido no grafo

| Output shape do tflite | Significado |
|------------------------|-------------|
| `[1, 84, 8400]` | **Cru** — sem NMS embutido. App precisa fazer NMS. |
| `[1, 300, 6]`   | **Com NMS embutido** — tflite ja devolve 300 detecoes filtradas (4 coords + class + conf) |

Script para checar qualquer tflite:

```python
from ai_edge_litert.interpreter import Interpreter
interp = Interpreter(model_path="modelo.tflite")
interp.allocate_tensors()
print("Output shape:", interp.get_output_details()[0]['shape'])
```

### Por que tirar NMS do grafo e a melhor estrategia para mobile

| | NMS no grafo (GPU) | NMS no app (CPU) |
|---|---|---|
| Inferencia GPU | Hibrido (~39 ms) ou falha | Plena (~9 ms) |
| NMS por si | ~0 ms (em GPU se desse) | ~1 ms em Kotlin |
| **Total** | ~39 ms ou falha | **~10 ms** |

Saldo: tirar 10 ops do lugar errado da ~4× speedup.

---

## Apendice C — Pipeline de quantizacao corrigido

### Localizacao

Projeto irmao em `../quant-modelos/`, scripts em `quant-modelos/scripts/`:
- `02_quantize_dynamic.py` — gera `*_int8_dr.tflite` (dynamic range)
- `03_quantize_full_int8.py` — gera `*_int8_full.tflite` (full int8 + COCO calib)

### Mudancas aplicadas

**Em `02_quantize_dynamic.py`** — substituido caminho via SavedModel pelo path
Ultralytics direto:

```python
# ANTES (deixava 3 ops orfas no grafo):
converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# DEPOIS (passa por ONNX, faz dead-code elimination):
model = YOLO(pt_name)
exported = model.export(
    format="tflite",
    imgsz=imgsz,
    int8=True,        # pesos int8
    nms=False,        # NMS fora do grafo
    end2end=False,    # grafo limpo
    batch=1,
)
```

**Em `03_quantize_full_int8.py`** — adicionados flags explicitos:

```python
exported = model.export(
    format="tflite",
    imgsz=imgsz,
    int8=True,
    data=str(calib_yaml),
    fraction=fraction,
    batch=1,
    nms=False,        # ADICIONADO
    end2end=False,    # ADICIONADO (yolo26n vinha True por default!)
)
```

### Comando completo de regeneracao

```bash
cd ../quant-modelos

# 1. Limpar caches (SavedModels foram gerados em ambientes diferentes,
#    causa de inconsistencia entre yolo11n e yolo26n)
rm -rf yolo11n_saved_model yolo26n_saved_model outputs/models/*

# 2. Rodar scripts corrigidos (~10-20 min total)
poetry run python scripts/02_quantize_dynamic.py
poetry run python scripts/03_quantize_full_int8.py

# 3. Copiar para o app
cp outputs/models/dynamic_range/*.tflite \
   outputs/models/full_int8/*.tflite \
   ../testes-yolo-apk-android/apk-yolo-flutter/example/android/app/src/main/assets/
```

### O caso DR persistente

Mesmo apos o refactor, `yolo11n_int8_dr.tflite` e `yolo26n_int8_dr.tflite`
continuam com **3 ops orfas** (`MatMul/Multiply/Transpose` com `batch=400`).
Confirmado que vem do exporter da Ultralytics 8.4.52 quando `int8=True` SEM
`data=...` (sem representative dataset). A calibracao full int8 (com `data=`)
elimina essas ops via dead-code elimination implicita.

**Solucao adotada:** marcar `useGpu: false` para os DR no `models.json`.
Eles rodam em CPU XNNPACK em ~37 ms sem problemas.

---

## Apendice D — Metodologia de investigacao

### Captura de logcat com filtro adequado

Tags essenciais para diagnostico:

```bash
adb logcat -v threadtime \
  ObjectDetector:V TfLiteGpuDelegate:V tflite:V TfLite:V \
  Interpreter:V Predictor:V YOLO:V \
  YOLOInstanceManager:V YOLOPlugin:V \
  AndroidRuntime:E '*:S' > /tmp/yolo_logs.txt
```

**Erro comum**: nao incluir `YOLOInstanceManager` e `YOLOPlugin` — foi por isso
que inicialmente nao vimos a `IllegalArgumentException` real (estava sendo
logada em `YOLOPlugin` que nao estava no filtro).

### Sinais reveladores nos logs

| Padrao | O que significa |
|--------|-----------------|
| `Replacing N out of M node(s) with delegate (TfLiteGpuDelegateV2)` | TFLite tentou colocar N ops em particoes GPU. N=M = unica particao. |
| `Created K GPU delegate kernels` | K kernels foram efetivamente compilados. **K=0 = falha total** mesmo com N>0. |
| `Replacing N out of M node(s) with delegate (TfLiteXNNPackDelegate)` | Quantas ops o XNNPACK pegou (fase CPU ou fallback). |
| `E YOLOPlugin: Error during prediction` + stacktrace | Predict falhou; capturado em `YOLOInstanceManager.predict:144`. |
| `TfLiteGpuDelegate Init: Batch size mismatch, expected 1 but got 400` | Op com batch>1 incompativel com GPU delegate v2. |
| `Created 1 GPU delegate kernels` em modelo "completo" | Sucesso ideal: 1 particao grande compilada. |

### Inspecao de tflite com `ai_edge_litert`

Instalacao:
```bash
pip3 install ai-edge-litert --break-system-packages
```

Scripts usados (salvos em `/tmp/` durante a investigacao):
- `inspect_tflite.py` — conta tensors, ops, identifica shapes com `batch != 1`
- `inspect_ops.py` — classifica cada op como int8/fp32/fp16 (revela quantizacao parcial)
- `check_nms.py` — heuristica para detectar NMS embutido via output shape

### Workflow de diagnostico (resumo)

1. `adb logcat -c` (limpar buffer)
2. Iniciar captura com filtro acima em background
3. Rodar benchmark no celular para o(s) modelo(s) suspeito(s)
4. Parar captura, segmentar log por modelo (procurar `Model loaded successfully`)
5. Para cada segmento:
   - Contar `Predict Total time` (= inferencias bem sucedidas)
   - Procurar `Error during prediction` (= GPU falhou silenciosamente)
   - Olhar `Created N GPU delegate kernels` (N=0 = falha total)
6. Cross-check com inspecao estatica do tflite (`ai_edge_litert`)

---

## Apendice E — Limites de hardware observados

### Convergencia empirica em ~9 ms

No S25+ (Adreno 830), todos os modelos nano GPU-compativeis convergem em
**~9 ms de inferencia pura**, independente da precisao ou arquitetura:

- `yolo11n_float16` GPU: 9.0 ms
- `yolo11n_int8_full` GPU: 9.0 ms
- `yolo26n_int8_full` GPU: 9.0 ms (apos refactor)

Se fosse compute-bound, esperariamos int8 ser ~2× mais rapido que fp16. Se
fosse memory-bound, fp16 (~2× menos dados que fp32) e int8 (~4× menos) teriam
tempos bem diferentes. **Nada disso acontece** → estamos batendo num teto
provavelmente de **memory bandwidth + driver scheduling granularity** da Adreno.

### Decomposicao do tempo end-to-end

O numero `avg_inf_ms` no JSON do benchmark e cronometrado em **Dart**, ao redor
da chamada `await yolo.predict(...)`. Isso inclui:

```
Dart predict() ──→ MethodChannel ──→ Kotlin handler ──→
   BitmapFactory.decodeByteArray (~3-5 ms)
   ↓
   YOLOInstanceManager.predict ──→ ObjectDetector.predict {
       Pre  (~25 ms): resize + RGB → ByteBuffer
       Inf  (~9 ms):  ← o "Predict Stage: Inference done" do log
       Post (~5 ms):  NMS + extracao de bboxes
   }
   ↓
   serializacao da resposta ──→ MethodChannel volta ──→ Dart
```

**Total end-to-end: ~40-50 ms** para o melhor caso (yolo11n_int8_full GPU).

Distincao importante:
- Log `Predict Stage: Inference done in X ms` = **X = TFLite puro** (~9 ms)
- JSON `avg_inf_ms` = **end-to-end Dart** (~40 ms)

Ambos sao corretos, medem coisas diferentes. Para reportar "tempo do modelo"
no estilo academico/Ultralytics, usar o do log. Para "tempo que o usuario ve",
usar o do JSON.

### Camadas de teto

Em ordem do mais facil de superar ao mais dificil:

1. **Pre/pos-processamento** (~30 ms fixos) — poderia ser reduzido com
   `Bitmap` pre-alocado, evitando `BitmapFactory.decodeByteArray` por inferencia,
   ou rodando preprocessing em RenderScript/GPU.
2. **Memory bandwidth** — ~76 GB/s no LPDDR5X do S25+. Para 150 MB de
   movimentacao por inferencia, ~2 ms de piso teorico.
3. **FLOPS** — Adreno 830 entrega ~2.5 TFLOPS fp16. Yolo11n tem ~6.5 GFLOPs/inf
   → ~2.6 ms de piso teorico. Estamos a 9 ms = ~28% do pico, longe disso ser
   o gargalo.
4. **Driver / kernel launch overhead** — desprezivel com 1 kernel.

### Como quebrar o teto (futuro)

| Estrategia | Ganho estimado | Custo |
|------------|----------------|-------|
| Reduzir input para 416×416 | ~2.4× | Re-treinar / re-exportar modelos; perde alguma acuracia |
| Pre-alocar bitmap, evitar decode por frame | ~5 ms de overhead | Refactor do plugin nativo |
| NNAPI / NPU | ~3-5× em phones com NPU | Codigo nativo + perde portabilidade (ver "O que NAO fizemos") |
| Batching | N× para batch processing | Nao se aplica a camera real-time |
