// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import 'package:ultralytics_yolo/utils/error_handler.dart';
import 'package:ultralytics_yolo/yolo_view.dart';

import '../../models/infraction_rule.dart';
import '../../models/model_descriptor.dart';
import '../../models/models.dart';
import '../../services/infraction_rules_storage.dart';
import '../../services/infraction_service.dart';
import '../../services/model_manager.dart';
import '../../services/model_registry.dart';
import '../../services/system_metrics_service.dart';

/// Controller que gerencia o estado e a logica de negocio da tela
/// de camera (inferencia em tempo real + regras de infracao + metricas).
class CameraInferenceController extends ChangeNotifier {
  // ===== Detection state =====
  int _detectionCount = 0;
  double _currentFps = 0.0;
  double _processingTimeMs = 0.0;
  List<YOLOResult> _lastDetections = const [];

  // ===== Threshold state =====
  double _confidenceThreshold = 0.5;
  double _iouThreshold = 0.45;
  int _numItemsThreshold = 300;
  SliderType _activeSlider = SliderType.none;

  // ===== Model state =====
  ModelDescriptor _selectedModel =
      ModelDescriptor.fromBundled(ModelRegistry.instance.defaultModel);
  bool _isModelLoading = false;
  String? _modelPath;
  String _loadingMessage = '';
  double _downloadProgress = 0.0;

  // ===== Camera state =====
  double _currentZoomLevel = 1.0;
  LensFacing _lensFacing = LensFacing.front;
  bool _isFrontCamera = false;

  // ===== Infraction state =====
  final InfractionService _infractionService = InfractionService();
  late InfractionRuleSet _activeRules;
  InfractionEvaluation _lastEval = InfractionEvaluation.empty;

  // ===== System metrics =====
  SystemMetrics? _lastSystemMetrics;
  StreamSubscription<SystemMetrics>? _metricsSub;

  // ===== Plugin controllers =====
  final _yoloController = YOLOViewController();
  late final ModelManager _modelManager;

  // ===== Lifecycle =====
  bool _isDisposed = false;
  Future<void>? _loadingFuture;

  // ===== Getters =====
  int get detectionCount => _detectionCount;
  double get currentFps => _currentFps;
  double get processingTimeMs => _processingTimeMs;
  List<YOLOResult> get lastDetections => _lastDetections;
  double get confidenceThreshold => _confidenceThreshold;
  double get iouThreshold => _iouThreshold;
  int get numItemsThreshold => _numItemsThreshold;
  SliderType get activeSlider => _activeSlider;
  ModelDescriptor get selectedModel => _selectedModel;
  bool get isModelLoading => _isModelLoading;
  String? get modelPath => _modelPath;
  String get loadingMessage => _loadingMessage;
  double get downloadProgress => _downloadProgress;
  double get currentZoomLevel => _currentZoomLevel;
  bool get isFrontCamera => _isFrontCamera;
  LensFacing get lensFacing => _lensFacing;
  YOLOViewController get yoloController => _yoloController;
  InfractionRuleSet get activeRules => _activeRules;
  InfractionEvaluation get lastEvaluation => _lastEval;
  SystemMetrics? get lastSystemMetrics => _lastSystemMetrics;

  CameraInferenceController() {
    _isFrontCamera = _lensFacing == LensFacing.front;
    _activeRules = _resolveRules(_selectedModel);
    _infractionService.primeWindow(_activeRules.absenceClasses);

    _modelManager = ModelManager(
      onDownloadProgress: (progress) {
        _downloadProgress = progress;
        notifyListeners();
      },
      onStatusUpdate: (message) {
        _loadingMessage = message;
        notifyListeners();
      },
    );
  }

  Future<void> initialize() async {
    _metricsSub = SystemMetricsService.instance.stream.listen((m) {
      if (_isDisposed) return;
      _lastSystemMetrics = m;
      notifyListeners();
    });
    await _loadModelForPlatform();
    _yoloController.setThresholds(
      confidenceThreshold: _confidenceThreshold,
      iouThreshold: _iouThreshold,
      numItemsThreshold: _numItemsThreshold,
    );
  }

  // ===== Detection callback =====
  void onDetectionResults(List<YOLOResult> results) {
    if (_isDisposed) return;

    _lastDetections = results;
    _detectionCount = results.length;

    final modelClasses = _selectedModel.classes?.toSet() ?? const <String>{};
    _lastEval = _infractionService.evaluate(
      results,
      _activeRules,
      modelClasses: modelClasses,
    );

    notifyListeners();
  }

  void onPerformanceMetrics(double fps) {
    if (_isDisposed) return;
    _currentFps = fps;
    notifyListeners();
  }

  /// Atualiza tempo de processamento por frame (ms). Recebido do
  /// callback `onPerformanceMetrics` do YOLOView.
  void onProcessingTime(double ms) {
    if (_isDisposed) return;
    _processingTimeMs = ms;
  }

  void onZoomChanged(double zoomLevel) {
    if (_isDisposed) return;

    if ((_currentZoomLevel - zoomLevel).abs() > 0.01) {
      _currentZoomLevel = zoomLevel;
      notifyListeners();
    }
  }

  // ===== Slider/threshold controls =====
  void toggleSlider(SliderType type) {
    if (_isDisposed) return;

    if (_activeSlider != type) {
      _activeSlider = _activeSlider == type ? SliderType.none : type;
      notifyListeners();
    }
  }

  void updateSliderValue(double value) {
    if (_isDisposed) return;

    bool changed = false;
    switch (_activeSlider) {
      case SliderType.numItems:
        final newValue = value.toInt();
        if (_numItemsThreshold != newValue) {
          _numItemsThreshold = newValue;
          _yoloController.setNumItemsThreshold(_numItemsThreshold);
          changed = true;
        }
        break;
      case SliderType.confidence:
        if ((_confidenceThreshold - value).abs() > 0.01) {
          _confidenceThreshold = value;
          _yoloController.setConfidenceThreshold(value);
          changed = true;
        }
        break;
      case SliderType.iou:
        if ((_iouThreshold - value).abs() > 0.01) {
          _iouThreshold = value;
          _yoloController.setIoUThreshold(value);
          changed = true;
        }
        break;
      default:
        break;
    }

    if (changed) {
      notifyListeners();
    }
  }

  void setZoomLevel(double zoomLevel) {
    if (_isDisposed) return;

    if ((_currentZoomLevel - zoomLevel).abs() > 0.01) {
      _currentZoomLevel = zoomLevel;
      _yoloController.setZoomLevel(zoomLevel);
      notifyListeners();
    }
  }

  void flipCamera() {
    if (_isDisposed) return;

    _isFrontCamera = !_isFrontCamera;
    _lensFacing = _isFrontCamera ? LensFacing.front : LensFacing.back;
    if (_isFrontCamera) _currentZoomLevel = 1.0;
    _yoloController.switchCamera();
    _infractionService.resetWindow();
    _infractionService.primeWindow(_activeRules.absenceClasses);
    notifyListeners();
  }

  void setLensFacing(LensFacing facing) {
    if (_isDisposed) return;

    if (_lensFacing != facing) {
      _lensFacing = facing;
      _isFrontCamera = facing == LensFacing.front;
      _yoloController.switchCamera();
      if (_isFrontCamera) {
        _currentZoomLevel = 1.0;
      }
      notifyListeners();
    }
  }

  // ===== Model selection =====
  void changeModel(ModelDescriptor model) {
    if (_isDisposed) return;

    if (!_isModelLoading && model != _selectedModel) {
      _selectedModel = model;
      _activeRules = _resolveRules(model);
      _infractionService.resetWindow();
      _infractionService.primeWindow(_activeRules.absenceClasses);
      _lastEval = InfractionEvaluation.empty;
      _lastDetections = const [];
      _detectionCount = 0;
      _loadModelForPlatform();
    }
  }

  // ===== Infraction rule overrides =====
  Future<void> applyRulesOverride(InfractionRuleSet rules) async {
    if (_isDisposed) return;
    await InfractionRulesStorage.instance.override(rules);
    _activeRules = rules;
    _infractionService.resetWindow();
    _infractionService.primeWindow(_activeRules.absenceClasses);
    notifyListeners();
  }

  Future<void> resetRulesToDefaults() async {
    if (_isDisposed) return;
    await InfractionRulesStorage.instance.reset(_selectedModel.key);
    _activeRules = _selectedModel.defaultRules;
    _infractionService.resetWindow();
    _infractionService.primeWindow(_activeRules.absenceClasses);
    notifyListeners();
  }

  InfractionRuleSet _resolveRules(ModelDescriptor model) {
    return InfractionRulesStorage.instance.ruleSetFor(
      model.key,
      model.defaultRules,
    );
  }

  // ===== Model loading =====
  Future<void> _loadModelForPlatform() async {
    if (_isDisposed) return;

    if (_loadingFuture != null) {
      await _loadingFuture;
      return;
    }

    _loadingFuture = _performModelLoading();
    try {
      await _loadingFuture;
    } finally {
      _loadingFuture = null;
    }
  }

  Future<void> _performModelLoading() async {
    if (_isDisposed) return;

    _isModelLoading = true;
    _loadingMessage = 'Carregando ${_selectedModel.label}...';
    _downloadProgress = 0.0;
    _detectionCount = 0;
    _currentFps = 0.0;
    _processingTimeMs = 0.0;
    _lastDetections = const [];
    _lastEval = InfractionEvaluation.empty;
    notifyListeners();

    try {
      final modelPath =
          await _modelManager.getDescriptorPath(_selectedModel);

      if (_isDisposed) return;

      _modelPath = modelPath;
      _isModelLoading = false;
      _loadingMessage = '';
      _downloadProgress = 0.0;
      notifyListeners();

      if (modelPath == null) {
        throw Exception('Falha ao carregar modelo ${_selectedModel.label}');
      }
    } catch (e) {
      if (_isDisposed) return;

      final error = YOLOErrorHandler.handleError(
        e,
        'Falha ao carregar modelo ${_selectedModel.label} (task ${_selectedModel.task.name})',
      );

      _isModelLoading = false;
      _loadingMessage = 'Erro: ${error.message}';
      _downloadProgress = 0.0;
      notifyListeners();
      rethrow;
    }
  }

  @override
  void dispose() {
    _isDisposed = true;
    _metricsSub?.cancel();
    _metricsSub = null;
    super.dispose();
  }
}
