import 'package:ultralytics_yolo/models/yolo_task.dart';

import 'infraction_rule.dart';
import 'models.dart';

/// Origem do modelo: bundled (asset embutido no APK) ou customizado
/// (arquivo .tflite escolhido pelo usuario via file_picker).
enum ModelSource { bundled, custom }

/// Descritor unificado de um modelo carregavel pela POC.
///
/// Encapsula `ModelInfo` (quando bundled) ou os metadados minimos de
/// um modelo custom escolhido em runtime, mais a lista de classes e
/// as regras default. `key` e usado como identificador estavel para
/// persistencia de overrides em SharedPreferences.
class ModelDescriptor {
  /// Identificador estavel:
  /// - bundled: `bundled:<filename>`
  /// - custom : `custom:<absolute path>`
  final String key;

  final String label;
  final YOLOTask task;
  final ModelSource source;

  /// Caminho que o YOLOView deve usar para carregar o modelo.
  /// Para bundled, e o nome do arquivo (sem extensao) -- o plugin
  /// resolve via assets nativos. Para custom, e o caminho absoluto.
  final String modelPath;

  /// `ModelInfo` original (so para bundled).
  final ModelInfo? bundled;

  /// Lista de classes do modelo. `null` significa "desconhecidas".
  final List<String>? classes;

  /// Regras default carregadas do manifest (bundled) ou vazias (custom).
  final InfractionRuleSet defaultRules;

  const ModelDescriptor({
    required this.key,
    required this.label,
    required this.task,
    required this.source,
    required this.modelPath,
    required this.bundled,
    required this.classes,
    required this.defaultRules,
  });

  factory ModelDescriptor.fromBundled(ModelInfo info) {
    final key = 'bundled:${info.file}';
    return ModelDescriptor(
      key: key,
      label: info.label,
      task: info.task,
      source: ModelSource.bundled,
      modelPath: info.name,
      bundled: info,
      classes: info.classes,
      defaultRules: info.defaultRules,
    );
  }

  factory ModelDescriptor.fromCustomFile({
    required String absolutePath,
    required String label,
    YOLOTask task = YOLOTask.detect,
    List<String>? classes,
  }) {
    final key = 'custom:$absolutePath';
    return ModelDescriptor(
      key: key,
      label: label,
      task: task,
      source: ModelSource.custom,
      modelPath: absolutePath,
      bundled: null,
      classes: classes,
      defaultRules: InfractionRuleSet.empty(key),
    );
  }

  /// Cria copia com classes atualizadas (usado quando o app aprende
  /// classes em runtime para um modelo custom).
  ModelDescriptor withClasses(List<String> learned) {
    return ModelDescriptor(
      key: key,
      label: label,
      task: task,
      source: source,
      modelPath: modelPath,
      bundled: bundled,
      classes: learned,
      defaultRules: defaultRules,
    );
  }

  @override
  bool operator ==(Object other) =>
      other is ModelDescriptor && other.key == key;

  @override
  int get hashCode => key.hashCode;
}
