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
              _SessionRecordControl(controller: controller),
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

/// Inicia / para gravacao de video MP4 (`Movies/POC_EPI/` no Android).
class _SessionRecordControl extends StatelessWidget {
  const _SessionRecordControl({required this.controller});

  final CameraInferenceController controller;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: controller,
      builder: (context, _) {
        final rec = controller.isSessionRecording;
        return Padding(
          padding: const EdgeInsets.only(left: 8),
          child: Material(
            color: Colors.black45,
            borderRadius: BorderRadius.circular(20),
            child: InkWell(
              onTap: controller.isModelLoading
                  ? null
                  : () => _toggle(context),
              borderRadius: BorderRadius.circular(20),
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      rec ? Icons.stop_circle : Icons.fiber_manual_record,
                      color: rec ? Colors.white : Colors.redAccent,
                      size: 22,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      rec ? 'Parar' : 'Gravar',
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w600,
                        fontSize: 13,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Future<void> _toggle(BuildContext context) async {
    if (controller.isSessionRecording) {
      final videoPath = await controller.stopSessionRecording();
      if (!context.mounted) return;
      final msg = videoPath != null && videoPath.isNotEmpty
          ? 'Gravação terminada.\nVídeo: $videoPath'
          : 'Gravação terminada.';
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
      return;
    }
    await controller.startSessionRecording();
    if (!context.mounted) return;
    final video = controller.sessionVideoTargetPath;
    final msg = video != null && video.isNotEmpty
        ? 'A gravar vídeo MP4.\n$video'
        : 'Não foi possível iniciar o vídeo neste dispositivo.';
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg),
        duration: const Duration(seconds: 5),
      ),
    );
  }
}
