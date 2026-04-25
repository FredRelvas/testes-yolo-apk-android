// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import '../../models/infraction_rule.dart';
import '../controllers/camera_inference_controller.dart';
import '../widgets/camera_inference_content.dart';
import '../widgets/camera_inference_overlay.dart';
import '../widgets/camera_logo_overlay.dart';
import '../widgets/camera_controls.dart';
import '../widgets/infraction_config_sheet.dart';
import '../widgets/threshold_slider.dart';

/// Tela de teste em tempo real com camera e deteccao YOLO.
///
/// Recursos:
/// - Camera ao vivo com bounding boxes coloridos (verde / vermelho)
/// - Banner de infracao quando regras sao disparadas
/// - Painel de metricas (FPS, ms, RAM, bateria)
/// - Selecao de modelo (bundled ou .tflite do dispositivo)
/// - Configuracao de regras de infracao via icone na AppBar
/// - Thresholds ajustaveis (confidence, IoU, max detections)
/// - Controles de camera (flip, zoom)
class CameraInferenceScreen extends StatefulWidget {
  const CameraInferenceScreen({super.key});

  @override
  State<CameraInferenceScreen> createState() => _CameraInferenceScreenState();
}

class _CameraInferenceScreenState extends State<CameraInferenceScreen> {
  late final CameraInferenceController _controller;
  int _rebuildKey = 0;

  @override
  void initState() {
    super.initState();
    _controller = CameraInferenceController();
    _controller.initialize().catchError((error) {
      if (mounted) {
        _showError('Model Loading Error', error.toString());
      }
    });
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Check if route is current (we've navigated back to this screen)
    final route = ModalRoute.of(context);
    if (route?.isCurrent == true) {
      // Force rebuild when navigating back to ensure camera restarts
      // The rebuild will create a new YOLOView which will automatically start the camera
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) {
          setState(() {
            _rebuildKey++;
          });
        }
      });
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isLandscape =
        MediaQuery.of(context).orientation == Orientation.landscape;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Camera em tempo real'),
        actions: [
          ListenableBuilder(
            listenable: _controller,
            builder: (context, _) => IconButton(
              icon: const Icon(Icons.rule),
              tooltip: 'Configurar regras de infracao',
              onPressed: _controller.isModelLoading
                  ? null
                  : () => _openRulesSheet(),
            ),
          ),
        ],
      ),
      body: ListenableBuilder(
        listenable: _controller,
        builder: (context, child) {
          return Stack(
            children: [
              CameraInferenceContent(
                key: ValueKey('camera_content_$_rebuildKey'),
                controller: _controller,
                rebuildKey: _rebuildKey,
              ),
              CameraInferenceOverlay(
                controller: _controller,
                isLandscape: isLandscape,
              ),
              CameraLogoOverlay(
                controller: _controller,
                isLandscape: isLandscape,
              ),
              CameraControls(
                currentZoomLevel: _controller.currentZoomLevel,
                isFrontCamera: _controller.isFrontCamera,
                activeSlider: _controller.activeSlider,
                onZoomChanged: _controller.setZoomLevel,
                onSliderToggled: _controller.toggleSlider,
                onCameraFlipped: _controller.flipCamera,
                isLandscape: isLandscape,
              ),
              ThresholdSlider(
                activeSlider: _controller.activeSlider,
                confidenceThreshold: _controller.confidenceThreshold,
                iouThreshold: _controller.iouThreshold,
                numItemsThreshold: _controller.numItemsThreshold,
                onValueChanged: _controller.updateSliderValue,
                isLandscape: isLandscape,
              ),
            ],
          );
        },
      ),
    );
  }

  Future<void> _openRulesSheet() async {
    final result = await showModalBottomSheet<InfractionRuleSet>(
      context: context,
      isScrollControlled: true,
      builder: (_) => InfractionConfigSheet(
        model: _controller.selectedModel,
        current: _controller.activeRules,
      ),
    );
    if (result == null || !mounted) return;
    if (result == _controller.selectedModel.defaultRules) {
      await _controller.resetRulesToDefaults();
    } else {
      await _controller.applyRulesOverride(result);
    }
  }

  void _showError(String title, String message) => showDialog(
    context: context,
    builder: (context) => AlertDialog(
      title: Text(title),
      content: Text(message),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('OK'),
        ),
      ],
    ),
  );
}
