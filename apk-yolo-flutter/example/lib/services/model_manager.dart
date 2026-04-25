// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'package:archive/archive.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/utils/map_converter.dart';
import 'package:ultralytics_yolo/config/channel_config.dart';
import '../models/model_descriptor.dart';
import '../models/models.dart';

/// Manages YOLO model loading, downloading, and caching.
///
/// This class handles:
/// - Checking for existing models in the app bundle
/// - Downloading models from the Ultralytics GitHub releases
/// - Extracting and caching models locally
/// - Platform-specific model path management
class ModelManager {
  /// Base URL for downloading model files from GitHub releases
  static const String _modelDownloadBaseUrl =
      'https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.0.0';

  static final MethodChannel _channel =
      ChannelConfig.createSingleImageChannel();

  /// Callback for download progress updates (0.0 to 1.0)
  final void Function(double progress)? onDownloadProgress;

  /// Callback for status message updates
  final void Function(String message)? onStatusUpdate;

  /// Creates a new ModelManager instance
  ///
  /// [onDownloadProgress] is called with progress updates during model downloads
  /// [onStatusUpdate] is called with status messages during model operations
  ModelManager({this.onDownloadProgress, this.onStatusUpdate});

  /// Gets the appropriate model path for the current platform and model type.
  Future<String?> getModelPath(ModelInfo model) async => Platform.isIOS
      ? _getIOSModelPath(model)
      : Platform.isAndroid
      ? _getAndroidModelPath(model)
      : null;

  /// Resolve um [ModelDescriptor] em um caminho que o YOLOView aceita.
  ///
  /// Para `ModelSource.bundled`, delega para [getModelPath]. Para
  /// `ModelSource.custom`, retorna o caminho absoluto direto (o
  /// arquivo ja foi copiado para `getApplicationDocumentsDirectory()`
  /// pelo `ModelPicker`).
  Future<String?> getDescriptorPath(ModelDescriptor descriptor) async {
    if (descriptor.source == ModelSource.custom) {
      final file = File(descriptor.modelPath);
      if (await file.exists()) return descriptor.modelPath;
      _updateStatus('Arquivo nao encontrado: ${descriptor.modelPath}');
      return null;
    }
    final bundled = descriptor.bundled;
    if (bundled == null) return null;
    return getModelPath(bundled);
  }

  /// Gets the iOS model path (.mlpackage format).
  Future<String?> _getIOSModelPath(ModelInfo model) async {
    _updateStatus('Checking for ${model.name} model...');
    try {
      final bundleCheck = await _checkModelExistsInBundle(model.name);
      if (bundleCheck['exists'] == true) return model.name;
    } catch (_) {}
    final dir = await getApplicationDocumentsDirectory();
    final modelDir = Directory('${dir.path}/${model.name}.mlpackage');
    if (await modelDir.exists()) {
      if (await File('${modelDir.path}/Manifest.json').exists()) {
        return modelDir.path;
      }
      await modelDir.delete(recursive: true);
    }
    _updateStatus('Downloading ${model.name} model...');
    return _downloadIOSModel(model);
  }

  /// Check if a model exists in the iOS bundle
  Future<Map<String, dynamic>> _checkModelExistsInBundle(
    String modelName,
  ) async {
    if (!Platform.isIOS) return {'exists': false};
    try {
      final result = await _channel.invokeMethod('checkModelExists', {
        'modelPath': modelName,
      });
      return MapConverter.convertToTypedMap(result);
    } catch (_) {
      return {'exists': false};
    }
  }

  /// Download iOS model (.mlpackage format) or extract from assets
  Future<String?> _downloadIOSModel(ModelInfo model) async {
    final dir = await getApplicationDocumentsDirectory();
    final modelDir = Directory('${dir.path}/${model.name}.mlpackage');
    if (await modelDir.exists()) return modelDir.path;
    try {
      final zipData = await rootBundle.load(
        'assets/models/${model.name}.mlpackage.zip',
      );
      return await _extractZip(
        zipData.buffer.asUint8List(),
        modelDir,
        model.name,
      );
    } catch (_) {}
    return await _downloadAndExtract(model, modelDir, '.mlpackage.zip');
  }

  /// Gets the Android model path (.tflite format)
  Future<String?> _getAndroidModelPath(ModelInfo model) async {
    _updateStatus('Checking for ${model.name} model...');
    final bundledName = model.file;

    // Check Android native assets first
    try {
      final result = await _channel.invokeMethod('checkModelExists', {
        'modelPath': bundledName,
      });
      if (result != null && result['exists'] == true) {
        return result['location'] == 'assets'
            ? bundledName
            : result['path'] as String;
      }
    } catch (_) {}

    // Check local storage
    final dir = await getApplicationDocumentsDirectory();
    final modelFile = File('${dir.path}/$bundledName');
    if (await modelFile.exists()) return modelFile.path;

    // Download if not found
    _updateStatus('Downloading ${model.name} model...');
    final bytes = await _downloadFile('$_modelDownloadBaseUrl/$bundledName');
    if (bytes != null && bytes.isNotEmpty) {
      await modelFile.writeAsBytes(bytes);
      return modelFile.path;
    }
    return null;
  }

  /// Helper method to download file with progress tracking
  Future<List<int>?> _downloadFile(String url) async {
    try {
      final client = http.Client();
      final request = await client.send(http.Request('GET', Uri.parse(url)));
      final contentLength = request.contentLength ?? 0;
      final bytes = <int>[];
      int downloadedBytes = 0;

      await for (final chunk in request.stream) {
        bytes.addAll(chunk);
        downloadedBytes += chunk.length;
        if (contentLength > 0) {
          onDownloadProgress?.call(downloadedBytes / contentLength);
        }
      }
      client.close();
      return bytes;
    } catch (_) {
      return null;
    }
  }

  /// Helper method to extract zip file
  Future<String?> _extractZip(
    List<int> bytes,
    Directory targetDir,
    String modelName,
  ) async {
    try {
      _updateStatus('Extracting model...');
      final archive = ZipDecoder().decodeBytes(bytes);
      await targetDir.create(recursive: true);
      String? prefix;
      if (archive.files.isNotEmpty) {
        final first = archive.files.first.name;
        if (first.contains('/') &&
            first.split('/').first.endsWith('.mlpackage')) {
          final topDir = first.split('/').first;
          if (archive.files.every(
            (f) => f.name.startsWith('$topDir/') || f.name == topDir,
          )) {
            prefix = '$topDir/';
          }
        }
      }
      for (final file in archive) {
        var filename = file.name;
        if (prefix != null) {
          if (filename.startsWith(prefix)) {
            filename = filename.substring(prefix.length);
          } else if (filename == prefix.replaceAll('/', '')) {
            continue;
          }
        }
        if (filename.isEmpty) continue;
        if (file.isFile) {
          final outputFile = File('${targetDir.path}/$filename');
          await outputFile.parent.create(recursive: true);
          await outputFile.writeAsBytes(file.content as List<int>);
        }
      }
      return targetDir.path;
    } catch (_) {
      if (await targetDir.exists()) {
        await targetDir.delete(recursive: true);
      }
      return null;
    }
  }

  /// Helper method to download and extract model
  Future<String?> _downloadAndExtract(
    ModelInfo model,
    Directory targetDir,
    String ext,
  ) async {
    final bytes = await _downloadFile(
      '$_modelDownloadBaseUrl/${model.name}$ext',
    );
    if (bytes == null) return null;
    return ext.contains('zip')
        ? await _extractZip(bytes, targetDir, model.name)
        : (await File(targetDir.path).writeAsBytes(bytes), targetDir.path).$2;
  }

  /// Updates the status message
  void _updateStatus(String message) => onStatusUpdate?.call(message);
}
