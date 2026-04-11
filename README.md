# YOLO Flutter - App Android com TFLite

App Flutter para rodar modelos YOLO convertidos em TFLite no celular Android, com suporte a multiplas inferencias (camera em tempo real, imagem unica e batch).

Baseado no plugin oficial [Ultralytics YOLO Flutter](https://github.com/ultralytics/yolo-flutter-app) (v0.2.0).

## Requisitos

- Flutter >= 3.32.1
- Dart >= 3.8.1
- Android SDK (API 35 para compilacao, minimo API 21)
- NDK 28.2.13676358
- JDK 17
- Dispositivo Android fisico (camera nao funciona no emulador)

## Como rodar

### 1. Instalar dependencias

```bash
cd example
flutter pub get
```

### 2. Conectar o celular Android via USB

Verificar se o dispositivo foi detectado:

```bash
flutter devices
```

### 3. Executar o app no celular

```bash
cd example
flutter run -d <DEVICE_ID>
```

Para filtrar logs indesejados no terminal (ex: spam de "Kumiho"):

```bash
flutter run -d <DEVICE_ID> 2>&1 | grep -v -i "kumiho"
```

Teclas uteis durante o `flutter run`:

- `r` -- hot reload (aplica mudancas rapido)
- `R` -- hot restart (reinicia o app)
- `q` -- sair

## Modelos TFLite

Os modelos ficam em `example/android/app/src/main/assets/`. Atualmente incluidos:

| Modelo | Tamanho | Descricao |
|--------|---------|-----------|
| `modelcoco_float32.tflite` | 9.5 MB | COCO dataset, precisao total |
| `modelcoco_float16.tflite` | 5 MB | COCO dataset, meia precisao |
| `exp_yolo26m_epi_negative_float32.tflite` | 78 MB | Modelo experimental EPI |

Para adicionar novos modelos, coloque o arquivo `.tflite` nessa mesma pasta.

### Como exportar seus proprios modelos

```python
from ultralytics import YOLO

# Deteccao
YOLO("yolo11n.pt").export(format="tflite")

# Segmentacao
YOLO("yolo11n-seg.pt").export(format="tflite")

# Classificacao
YOLO("yolo11n-cls.pt").export(format="tflite")

# Pose
YOLO("yolo11n-pose.pt").export(format="tflite")
```

## Tarefas YOLO suportadas

| Tarefa | Classe | Descricao |
|--------|--------|-----------|
| `detect` | `ObjectDetector` | Deteccao de objetos com bounding boxes |
| `segment` | `Segmenter` | Segmentacao por instancia (mascaras) |
| `classify` | `Classifier` | Classificacao de imagens |
| `pose` | `PoseEstimator` | Estimativa de pose (keypoints) |
| `obb` | `ObbDetector` | Bounding boxes orientados (rotacionados) |

## Telas do app exemplo

- **Home** - Selecao de funcionalidade
- **Camera Inference** - Deteccao em tempo real pela camera (ate 30 FPS)
- **Single Image** - Inferencia em uma unica imagem da galeria
- **Batch Inference** - Multiplas inferencias de uma vez
- **Benchmark** - Testes de performance do modelo

## Estrutura de pastas

```
apk-yolo-flutter/
│
├── lib/                          # Plugin YOLO Flutter (nucleo)
│   ├── yolo.dart                 #   API principal - classe YOLO
│   ├── yolo_view.dart            #   Widget para camera em tempo real
│   ├── yolo_instance_manager.dart#   Gerencia multiplas instancias YOLO
│   ├── yolo_streaming_config.dart#   Config de streaming (FPS, skip frames)
│   ├── yolo_performance_metrics.dart # Metricas de performance
│   ├── ultralytics_yolo.dart     #   Exporta toda a API publica
│   ├── core/
│   │   ├── yolo_inference.dart   #   Executa inferencia e processa resultados
│   │   └── yolo_model_manager.dart#  Carrega/troca modelos TFLite
│   ├── models/
│   │   ├── yolo_task.dart        #   Enum das 5 tarefas (detect, segment, etc.)
│   │   ├── yolo_result.dart      #   Classes de resultado (boxes, masks, keypoints)
│   │   └── yolo_exceptions.dart  #   Excecoes customizadas
│   ├── platform/
│   │   ├── yolo_platform_interface.dart  # Interface abstrata
│   │   └── yolo_platform_impl.dart       # Implementacao via MethodChannel
│   ├── widgets/
│   │   ├── yolo_controller.dart  #   Controla camera, thresholds, zoom
│   │   ├── yolo_overlay.dart     #   Desenha bounding boxes sobre a camera
│   │   └── yolo_controls.dart    #   Sliders e botoes de controle
│   ├── config/
│   │   └── channel_config.dart   #   Nomes dos canais de comunicacao nativo
│   └── utils/
│       ├── map_converter.dart    #   Conversao de tipos (dynamic -> typed maps)
│       ├── error_handler.dart    #   Tratamento de erros de plataforma
│       └── logger.dart           #   Log de debug
│
├── android/                      # Codigo nativo Android (Kotlin + C++)
│   ├── build.gradle              #   Config Gradle (TFLite 1.4.0, GPU delegate)
│   └── src/main/
│       ├── AndroidManifest.xml   #   Permissoes (CAMERA, INTERNET)
│       ├── cpp/
│       │   ├── CMakeLists.txt    #   Build CMake para lib nativa
│       │   └── native-lib.cpp    #   NMS (Non-Max Suppression) em C++ via JNI
│       └── kotlin/com/ultralytics/yolo/
│           ├── YOLOPlugin.kt     #   Entry point do plugin Flutter
│           ├── YOLO.kt           #   Motor de inferencia TFLite
│           ├── YOLOView.kt       #   View nativa com camera
│           ├── YOLOPlatformView.kt       # Platform view
│           ├── YOLOPlatformViewFactory.kt# Factory
│           ├── YOLOInstanceManager.kt    # Gerencia instancias
│           ├── ObjectDetector.kt #   Inferencia de deteccao
│           ├── Classifier.kt     #   Inferencia de classificacao
│           ├── Segmenter.kt      #   Inferencia de segmentacao
│           ├── PoseEstimator.kt  #   Inferencia de pose
│           ├── ObbDetector.kt    #   Inferencia de OBB
│           ├── Predictor.kt      #   Interface base de predicao
│           ├── ImageUtils.kt     #   Processamento de imagem
│           ├── GeometryUtils.kt  #   Calculos geometricos
│           ├── Utils.kt          #   Utilitarios gerais
│           ├── YOLOFileUtils.kt  #   I/O de arquivos e modelos
│           ├── YOLOResult.kt     #   Estrutura de resultado
│           ├── YOLOStreamConfig.kt #  Config de streaming
│           ├── YOLOTask.kt       #   Enum de tarefas
│           └── OBB.kt            #   Estrutura de bounding box orientado
│
├── example/                      # App exemplo (o que voce roda)
│   ├── pubspec.yaml              #   Dependencias do app
│   ├── main.dart                 #   Entry point alternativo
│   ├── lib/
│   │   ├── main.dart             #   Entry point principal
│   │   ├── models/               #   Data classes (benchmark, modelos)
│   │   ├── services/
│   │   │   └── model_manager.dart#   Gerencia ciclo de vida dos modelos
│   │   ├── presentation/
│   │   │   ├── screens/          #   5 telas (home, camera, single, batch, benchmark)
│   │   │   ├── controllers/      #   Controllers de estado
│   │   │   └── widgets/          #   10 widgets de UI
│   │   └── utils/
│   │       └── detection_eval_metrics.dart  # Metricas de avaliacao
│   ├── assets/                   #   Logos e imagens do app
│   └── android/
│       └── app/
│           ├── build.gradle      #   Config de build do app
│           ├── proguard-rules.pro#   Regras ProGuard (protege TFLite/JNI)
│           └── src/main/
│               ├── AndroidManifest.xml
│               └── assets/       #   MODELOS TFLITE FICAM AQUI
│
├── pubspec.yaml                  # Definicao do plugin Flutter
├── analysis_options.yaml         # Regras de lint do Dart
├── LICENSE                       # AGPL-3.0
├── .gitignore                    # Arquivos ignorados pelo Git
│
├── doc/                          # Documentacao de referencia (9 guias)
├── test/                         # Testes unitarios do plugin (10 arquivos)
├── .github/                      # CI/CD do repo original (workflows)
└── CHANGELOG.md                  # Historico de versoes
```

## O que foi modificado do repositorio original

Este projeto e um fork do [ultralytics/yolo-flutter-app](https://github.com/ultralytics/yolo-flutter-app) com as seguintes mudancas:

### Removido (limpeza de ~4.1 GB)

| O que | Tamanho | Motivo |
|-------|---------|--------|
| `ios/` | 332 KB | Implementacao iOS (Swift/CoreML) -- foco apenas em Android |
| `example/ios/` | 308 KB | Config Xcode do app exemplo -- nao necessario |
| `local_inference/` | 2.2 GB | Ambiente Python + dataset COCO + modelos de teste -- usado apenas para validacao local no Mac |
| `example/build/` | 1.8 GB | Artefatos de compilacao (regenerado automaticamente) |
| `example/.dart_tool/` | 70 MB | Cache Dart (regenerado por `flutter pub get`) |
| `android/.cxx/` | ~MB | Cache CMake (regenerado pelo Gradle) |

### Alterado

- `pubspec.yaml` -- Removido bloco de plataforma `ios:` (linhas 44-45 do original) para evitar warnings do Flutter

## Uso basico no codigo

### Inferencia em imagem unica

```dart
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

final yolo = YOLO(
  modelPath: 'assets/models/modelcoco_float32.tflite',
  task: YOLOTask.detect,
);

await yolo.loadModel();
final results = await yolo.predict(imageBytes);

for (final r in results) {
  print('${r.className}: ${(r.confidence * 100).toStringAsFixed(1)}%');
}
```

### Camera em tempo real

```dart
YOLOView(
  modelPath: 'modelcoco_float32.tflite',
  task: YOLOTask.detect,
  onResult: (results) {
    print('Detectados: ${results.length} objetos');
  },
)
```

## Licenca

AGPL-3.0 -- veja [LICENSE](LICENSE) para detalhes.
