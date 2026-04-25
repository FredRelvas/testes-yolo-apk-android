// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:ultralytics_yolo/models/yolo_task.dart';

import 'infraction_rule.dart';

/// Metadata describing a single TFLite model available to the app.
///
/// Values are produced by [ModelRegistry] from the native asset listing
/// (plus optional overrides declared in `assets/models.json`).
class ModelInfo {
  /// Filename as it appears in `android/app/src/main/assets/` (includes `.tflite`).
  final String file;

  /// Bare name without extension — used for logs, CSV filenames, etc.
  final String name;

  /// Friendly label shown in the UI. Defaults to [name] when no manifest entry exists.
  final String label;

  /// YOLO task the model was trained for. Defaults to `detect` when unspecified.
  final YOLOTask task;

  /// Whether this model should appear in benchmark/batch screens.
  final bool benchmark;

  /// Whether this is the initial selection on app startup.
  final bool isDefault;

  /// Lista ordenada de classes que o modelo retorna (index = classIndex).
  /// `null` quando o manifest nao declara -- POC vai aprender em runtime.
  final List<String>? classes;

  /// Regras de infracao default declaradas no `models.json`.
  /// Vazias quando o manifest nao declara.
  final InfractionRuleSet defaultRules;

  const ModelInfo({
    required this.file,
    required this.name,
    required this.label,
    required this.task,
    required this.benchmark,
    required this.isDefault,
    required this.classes,
    required this.defaultRules,
  });

  /// Backward-compatible alias — older code used `modelName` as the name w/o extension.
  String get modelName => name;

  @override
  bool operator ==(Object other) => other is ModelInfo && other.file == file;

  @override
  int get hashCode => file.hashCode;

  @override
  String toString() => 'ModelInfo($file, task=${task.name})';
}

enum SliderType { none, numItems, confidence, iou }
