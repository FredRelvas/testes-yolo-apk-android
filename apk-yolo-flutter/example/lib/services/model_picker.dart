import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

import '../models/model_descriptor.dart';

/// Permite ao usuario carregar um arquivo `.tflite` arbitrario do
/// armazenamento do dispositivo.
///
/// Para garantir um caminho estavel entre runs, o arquivo escolhido
/// e copiado para `getApplicationDocumentsDirectory()/custom_models/`
/// (se ainda nao estiver la).
class ModelPicker {
  ModelPicker._();

  static const _customModelsDir = 'custom_models';

  /// Abre o file picker para `.tflite`. Retorna `null` se o usuario
  /// cancelar. Em caso de erro, propaga a excecao.
  static Future<ModelDescriptor?> pickFromDevice() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['tflite'],
      allowMultiple: false,
      withData: false,
    );
    if (result == null || result.files.isEmpty) return null;

    final picked = result.files.single;
    final sourcePath = picked.path;
    if (sourcePath == null) {
      throw StateError('FilePicker did not return a path for ${picked.name}');
    }

    final destPath = await _ensureLocalCopy(sourcePath, picked.name);
    final label = p.basenameWithoutExtension(picked.name);
    return ModelDescriptor.fromCustomFile(
      absolutePath: destPath,
      label: label,
    );
  }

  static Future<String> _ensureLocalCopy(
    String sourcePath,
    String fileName,
  ) async {
    final docsDir = await getApplicationDocumentsDirectory();
    final targetDir = Directory(p.join(docsDir.path, _customModelsDir));
    if (!await targetDir.exists()) {
      await targetDir.create(recursive: true);
    }
    final targetPath = p.join(targetDir.path, fileName);

    // Se ja existe e tem o mesmo conteudo, reaproveita.
    final src = File(sourcePath);
    final dst = File(targetPath);
    if (await dst.exists()) {
      try {
        final srcLen = await src.length();
        final dstLen = await dst.length();
        if (srcLen == dstLen) return targetPath;
      } catch (e) {
        debugPrint('ModelPicker: length check failed ($e); will overwrite.');
      }
    }
    await src.copy(targetPath);
    return targetPath;
  }
}
