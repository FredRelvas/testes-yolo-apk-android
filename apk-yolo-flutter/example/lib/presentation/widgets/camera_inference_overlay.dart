// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import '../../models/models.dart';
import '../controllers/camera_inference_controller.dart';
import 'infraction_alert_banner.dart';
import 'metrics_panel.dart';
import 'model_selector_chip.dart';
import 'threshold_pill.dart';

/// Overlay superior da tela de camera. Linha de cima: chip do modelo
/// (esquerda) e painel de metricas (direita). Abaixo: banner de
/// infracao (so visivel quando ha) + threshold pill ativo.
class CameraInferenceOverlay extends StatelessWidget {
  const CameraInferenceOverlay({
    super.key,
    required this.controller,
    required this.isLandscape,
  });

  final CameraInferenceController controller;
  final bool isLandscape;

  @override
  Widget build(BuildContext context) {
    final eval = controller.lastEvaluation;
    return Positioned(
      top: MediaQuery.of(context).padding.top + (isLandscape ? 8 : 16),
      left: isLandscape ? 8 : 16,
      right: isLandscape ? 8 : 16,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Flexible(
                child: ModelSelectorChip(
                  current: controller.selectedModel,
                  isLoading: controller.isModelLoading,
                  onSelected: controller.changeModel,
                ),
              ),
              const Spacer(),
              MetricsPanel(
                fps: controller.currentFps,
                processingTimeMs: controller.processingTimeMs,
                detectionCount: controller.detectionCount,
                metrics: controller.lastSystemMetrics,
              ),
            ],
          ),
          SizedBox(height: isLandscape ? 8 : 12),
          InfractionAlertBanner(classes: eval.offendingClasses),
          const SizedBox(height: 8),
          _buildThresholdPills(),
        ],
      ),
    );
  }

  Widget _buildThresholdPills() {
    if (controller.activeSlider == SliderType.confidence) {
      return ThresholdPill(
        label:
            'CONFIDENCE THRESHOLD: ${controller.confidenceThreshold.toStringAsFixed(2)}',
      );
    } else if (controller.activeSlider == SliderType.iou) {
      return ThresholdPill(
        label: 'IOU THRESHOLD: ${controller.iouThreshold.toStringAsFixed(2)}',
      );
    } else if (controller.activeSlider == SliderType.numItems) {
      return ThresholdPill(label: 'ITEMS MAX: ${controller.numItemsThreshold}');
    }
    return const SizedBox.shrink();
  }
}
