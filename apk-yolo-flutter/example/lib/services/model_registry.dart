// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/config/channel_config.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';

import '../models/infraction_rule.dart';
import '../models/models.dart';

/// Dynamically discovered list of TFLite models bundled with the app.
///
/// Call [ModelRegistry.load] once during app startup (before `runApp`). After
/// that, consumers read [instance] synchronously.
///
/// Discovery rules:
///   1. Native side lists every `*.tflite` file in `android/app/src/main/assets/`.
///   2. If `assets/models.json` exists, its entries override defaults per filename.
///   3. Files without a manifest entry fall back to `task=detect`, `benchmark=true`,
///      label derived from the filename.
class ModelRegistry {
  ModelRegistry._(this.all);

  static ModelRegistry? _instance;

  /// Loaded registry. Access only after [load] has completed.
  static ModelRegistry get instance {
    final i = _instance;
    if (i == null) {
      throw StateError(
        'ModelRegistry.load() must be called before accessing instance.',
      );
    }
    return i;
  }

  /// All models bundled in the app.
  final List<ModelInfo> all;

  /// Subset flagged for benchmark/batch screens (defaults to all).
  List<ModelInfo> get benchmarkModels =>
      all.where((m) => m.benchmark).toList(growable: false);

  /// Entry marked `default: true` in `models.json`, or the first model.
  ModelInfo get defaultModel =>
      all.firstWhere((m) => m.isDefault, orElse: () => all.first);

  /// Loads the registry from native assets. Safe to call only once.
  static Future<void> load() async {
    final channel = ChannelConfig.createSingleImageChannel();
    List<String> tfliteFiles = const [];
    String? manifestJson;

    try {
      final result = await channel.invokeMethod<dynamic>('listModels');
      if (result is Map) {
        final files = result['files'];
        if (files is List) {
          tfliteFiles = files.whereType<String>().toList();
        }
        final manifest = result['manifest'];
        if (manifest is String && manifest.isNotEmpty) {
          manifestJson = manifest;
        }
      }
    } on MissingPluginException catch (e) {
      debugPrint('ModelRegistry: listModels not implemented on platform ($e)');
    } catch (e) {
      debugPrint('ModelRegistry: failed to list models ($e)');
    }

    final overrides = _parseManifest(manifestJson);
    final models = _merge(tfliteFiles, overrides);

    if (models.isEmpty) {
      debugPrint('ModelRegistry: no .tflite files discovered in assets.');
    }

    _instance = ModelRegistry._(models);
  }

  static Map<String, _ManifestEntry> _parseManifest(String? json) {
    if (json == null) return const {};
    try {
      final decoded = jsonDecode(json);
      if (decoded is! List) return const {};
      final map = <String, _ManifestEntry>{};
      for (final item in decoded) {
        if (item is! Map) continue;
        final file = item['file'];
        if (file is! String || file.isEmpty) continue;
        final rawClasses = item['classes'];
        final classes = rawClasses is List
            ? rawClasses.whereType<String>().toList(growable: false)
            : null;
        final rawRules = item['defaultInfractionRules'];
        map[file] = _ManifestEntry(
          label: item['label'] as String?,
          task: _parseTask(item['task']),
          benchmark: item['benchmark'] as bool?,
          isDefault: item['default'] as bool?,
          classes: classes,
          defaultRules: rawRules is Map
              ? Map<String, dynamic>.from(rawRules)
              : null,
        );
      }
      return map;
    } catch (e) {
      debugPrint('ModelRegistry: invalid models.json ($e)');
      return const {};
    }
  }

  static YOLOTask? _parseTask(dynamic raw) {
    if (raw is! String) return null;
    for (final t in YOLOTask.values) {
      if (t.name == raw) return t;
    }
    return null;
  }

  static List<ModelInfo> _merge(
    List<String> files,
    Map<String, _ManifestEntry> overrides,
  ) {
    // Sort for deterministic UI ordering, but keep manifest-declared order first.
    final manifestOrder = overrides.keys.toList();
    final extras = files.where((f) => !overrides.containsKey(f)).toList()
      ..sort();
    final ordered = [
      ...manifestOrder.where(files.contains),
      ...extras,
    ];

    return [
      for (final file in ordered)
        _buildInfo(file, overrides[file]),
    ];
  }

  static ModelInfo _buildInfo(String file, _ManifestEntry? entry) {
    final name = file.toLowerCase().endsWith('.tflite')
        ? file.substring(0, file.length - '.tflite'.length)
        : file;
    final modelKey = 'bundled:$file';
    return ModelInfo(
      file: file,
      name: name,
      label: entry?.label ?? name,
      task: entry?.task ?? YOLOTask.detect,
      benchmark: entry?.benchmark ?? true,
      isDefault: entry?.isDefault ?? false,
      classes: entry?.classes,
      defaultRules:
          InfractionRuleSet.fromManifest(modelKey, entry?.defaultRules),
    );
  }
}

class _ManifestEntry {
  final String? label;
  final YOLOTask? task;
  final bool? benchmark;
  final bool? isDefault;
  final List<String>? classes;
  final Map<String, dynamic>? defaultRules;

  const _ManifestEntry({
    required this.label,
    required this.task,
    required this.benchmark,
    required this.isDefault,
    required this.classes,
    required this.defaultRules,
  });
}
