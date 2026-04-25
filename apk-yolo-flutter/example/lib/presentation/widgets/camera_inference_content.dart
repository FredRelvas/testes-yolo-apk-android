// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import '../controllers/camera_inference_controller.dart';
import 'epi_overlay.dart';

/// Main content widget that handles the camera view and loading states.
/// Renderiza o YOLOView com `showOverlays: false` e desenha um
/// [EpiOverlay] customizado por cima para pintar bboxes em verde
/// (OK) ou vermelho (infracao).
class CameraInferenceContent extends StatelessWidget {
  const CameraInferenceContent({
    super.key,
    required this.controller,
    this.rebuildKey = 0,
  });

  final CameraInferenceController controller;
  final int rebuildKey;

  @override
  Widget build(BuildContext context) {
    if (controller.modelPath != null && !controller.isModelLoading) {
      return Stack(
        fit: StackFit.expand,
        children: [
          YOLOView(
            key: ValueKey(
              'yolo_view_${controller.modelPath}_${controller.selectedModel.task.name}_$rebuildKey',
            ),
            controller: controller.yoloController,
            modelPath: controller.modelPath!,
            task: controller.selectedModel.task,
            streamingConfig: const YOLOStreamingConfig.minimal(),
            showOverlays: false,
            onResult: controller.onDetectionResults,
            onPerformanceMetrics: (metrics) {
              controller.onPerformanceMetrics(metrics.fps);
              controller.onProcessingTime(metrics.processingTimeMs);
            },
            onZoomChanged: controller.onZoomChanged,
            lensFacing: controller.lensFacing,
          ),
          EpiOverlay(
            detections: controller.lastDetections,
            redBoxes: controller.lastEvaluation.redBoxes,
          ),
        ],
      );
    } else if (controller.isModelLoading) {
      return Container(
        color: Colors.black,
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(color: Colors.white),
              const SizedBox(height: 16),
              Text(
                controller.loadingMessage,
                style: const TextStyle(color: Colors.white),
              ),
              if (controller.downloadProgress > 0) ...[
                const SizedBox(height: 12),
                SizedBox(
                  width: 200,
                  child: LinearProgressIndicator(
                    value: controller.downloadProgress,
                    color: Colors.white,
                    backgroundColor: Colors.white24,
                  ),
                ),
              ],
            ],
          ),
        ),
      );
    } else {
      return const Center(
        child: Text('Nenhum modelo carregado',
            style: TextStyle(color: Colors.white)),
      );
    }
  }
}
