import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:ultralytics_yolo/yolo.dart';

import '../../models/benchmark_model_result.dart';
import '../../models/model_descriptor.dart';
import '../../models/models.dart';
import '../../services/model_manager.dart';
import '../../services/model_registry.dart';
import '../../services/system_metrics_service.dart';

/// Tela de benchmark comparativo entre modelos. Roda cada modelo
/// selecionado por N segundos e gera um relatorio com fps, inf ms,
/// mA, mAh e RAM media. Tres fases internas: setup -> running -> report.
class BenchmarkScreen extends StatefulWidget {
  const BenchmarkScreen({super.key});

  @override
  State<BenchmarkScreen> createState() => _BenchmarkScreenState();
}

enum _Phase { setup, running, report }

enum _Step { idle, loading, warmup, stabilizing, collecting, finalizing }

enum _AcceleratorMode { gpuOnly, cpuOnly, both }

/// Uma execucao individual do benchmark: um modelo rodando em um modo
/// especifico (GPU ou CPU). No modo `_AcceleratorMode.both`, cada modelo
/// gera duas execucoes (uma com `useGpu: true` e outra com `false`).
class _BenchmarkRun {
  final ModelInfo model;
  final bool useGpu;
  const _BenchmarkRun(this.model, this.useGpu);
}

class _BenchmarkScreenState extends State<BenchmarkScreen> {
  // -------- Setup --------
  late final List<ModelInfo> _availableModels;
  late final Set<String> _selectedFiles;
  int _durationSec = 30;
  static const List<int> _durationOptions = [15, 30, 60];
  _AcceleratorMode _acceleratorMode = _AcceleratorMode.gpuOnly;

  // -------- Run state --------
  _Phase _phase = _Phase.setup;
  bool _isRunning = false;
  int _currentModelIdx = 0;
  int _totalModels = 0;
  String _currentModelLabel = '';
  bool _currentUseGpu = true;
  _Step _currentStep = _Step.idle;
  DateTime? _stepStartTime;
  int _stepDurationSec = 0;
  String? _errorMessage;

  // -------- Live metrics --------
  int? _liveMa;
  double _liveFps = 0.0;
  final List<double> _recentInfTimes = [];
  static const int _liveFpsWindow = 30;
  Timer? _uiTickTimer;
  StreamSubscription<SystemMetrics>? _liveMetricsSub;

  // -------- Results --------
  final List<BenchmarkModelResult> _results = [];

  // -------- Shared resources --------
  final ModelManager _modelManager = ModelManager();
  Uint8List? _syntheticBytes;

  @override
  void initState() {
    super.initState();
    _availableModels = ModelRegistry.instance.all;
    _selectedFiles = _availableModels.map((m) => m.file).toSet();
  }

  @override
  void dispose() {
    _isRunning = false;
    _uiTickTimer?.cancel();
    _liveMetricsSub?.cancel();
    super.dispose();
  }

  // ===========================================================
  // Synthetic bitmap (640x640 PNG, gerado uma unica vez)
  // ===========================================================
  Future<Uint8List> _generateSyntheticBitmap() async {
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder, const Rect.fromLTWH(0, 0, 640, 640));
    canvas.drawRect(
      const Rect.fromLTWH(0, 0, 640, 640),
      Paint()..color = const Color(0xFF808080),
    );
    // Formas com contraste para o modelo nao ficar com input uniforme.
    canvas.drawCircle(
      const Offset(320, 320),
      150,
      Paint()..color = const Color(0xFFCCCCCC),
    );
    canvas.drawRect(
      const Rect.fromLTWH(100, 100, 150, 150),
      Paint()..color = const Color(0xFF404040),
    );
    canvas.drawRect(
      const Rect.fromLTWH(420, 420, 120, 120),
      Paint()..color = const Color(0xFFE8E8E8),
    );
    final picture = recorder.endRecording();
    final image = await picture.toImage(640, 640);
    final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    image.dispose();
    return byteData!.buffer.asUint8List();
  }

  // ===========================================================
  // Benchmark execution
  // ===========================================================

  /// Expande a lista de modelos selecionados em runs concretas
  /// considerando o modo de acelerador.
  List<_BenchmarkRun> _buildRuns(List<ModelInfo> selected) {
    final runs = <_BenchmarkRun>[];
    for (final m in selected) {
      switch (_acceleratorMode) {
        case _AcceleratorMode.gpuOnly:
          runs.add(_BenchmarkRun(m, true));
        case _AcceleratorMode.cpuOnly:
          runs.add(_BenchmarkRun(m, false));
        case _AcceleratorMode.both:
          runs.add(_BenchmarkRun(m, true));
          runs.add(_BenchmarkRun(m, false));
      }
    }
    return runs;
  }

  Future<void> _startBenchmark() async {
    final selected = _availableModels
        .where((m) => _selectedFiles.contains(m.file))
        .toList(growable: false);
    if (selected.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Selecione ao menos um modelo')),
      );
      return;
    }

    final runs = _buildRuns(selected);

    setState(() {
      _phase = _Phase.running;
      _isRunning = true;
      _results.clear();
      _totalModels = runs.length;
      _currentModelIdx = 0;
      _errorMessage = null;
    });

    _liveMetricsSub = SystemMetricsService.instance.stream.listen((m) {
      if (mounted) setState(() => _liveMa = m.currentMa);
    });

    _uiTickTimer = Timer.periodic(const Duration(milliseconds: 500), (_) {
      if (!mounted) return;
      double fps = 0.0;
      if (_recentInfTimes.isNotEmpty) {
        final avgMs = _recentInfTimes.reduce((a, b) => a + b) /
            _recentInfTimes.length;
        if (avgMs > 0) fps = 1000.0 / avgMs;
      }
      setState(() => _liveFps = fps);
    });

    _syntheticBytes ??= await _generateSyntheticBitmap();

    try {
      for (var i = 0; i < runs.length; i++) {
        if (!_isRunning) break;
        if (mounted) {
          setState(() {
            _currentModelIdx = i;
            _currentModelLabel = runs[i].model.label;
            _currentUseGpu = runs[i].useGpu;
          });
        }
        await _runOneModel(runs[i].model, useGpu: runs[i].useGpu);
      }
    } catch (e) {
      if (mounted) setState(() => _errorMessage = 'Erro: $e');
    } finally {
      _uiTickTimer?.cancel();
      await _liveMetricsSub?.cancel();
      _uiTickTimer = null;
      _liveMetricsSub = null;
      if (mounted) {
        setState(() {
          _phase = _Phase.report;
          _isRunning = false;
          _currentStep = _Step.idle;
        });
      }
    }
  }

  Future<void> _runOneModel(ModelInfo info, {required bool useGpu}) async {
    final descriptor = ModelDescriptor.fromBundled(info);

    // 1. Loading
    _setStep(_Step.loading, 0);
    SystemMetricsService.instance.resetSession();
    final path = await _modelManager.getDescriptorPath(descriptor);
    if (path == null) {
      throw Exception('Caminho nao encontrado para ${info.label}');
    }
    if (!_isRunning) return;

    final yolo = YOLO(
      modelPath: path,
      task: info.task,
      useGpu: useGpu,
      useMultiInstance: true,
    );
    await yolo.loadModel();
    if (!_isRunning) {
      await yolo.dispose();
      return;
    }

    try {
      // 2. Warmup (5s rodando inferencias para estabilizar JIT + GPU shaders)
      _setStep(_Step.warmup, 5);
      _recentInfTimes.clear();
      final warmupEnd = DateTime.now().add(const Duration(seconds: 5));
      while (DateTime.now().isBefore(warmupEnd) && _isRunning) {
        try {
          final sw = Stopwatch()..start();
          await yolo.predict(_syntheticBytes!, confidenceThreshold: 0.25);
          _pushRecentTime(sw.elapsedMicroseconds / 1000.0);
        } catch (_) {/* ignora erro individual de inferencia */}
      }
      if (!_isRunning) return;

      // 3. Estabilizacao (5s sem inferencia para caches assentarem)
      _setStep(_Step.stabilizing, 5);
      final stabEnd = DateTime.now().add(const Duration(seconds: 5));
      while (DateTime.now().isBefore(stabEnd) && _isRunning) {
        await Future.delayed(const Duration(milliseconds: 200));
      }
      if (!_isRunning) return;

      // 4. Coleta
      _setStep(_Step.collecting, _durationSec);
      _recentInfTimes.clear();

      final maReadings = <int>[];
      final ramReadings = <double>[];
      final infTimes = <double>[];
      double? collectStartMah;
      double? collectEndMah;

      final collectSub =
          SystemMetricsService.instance.stream.listen((m) {
        if (m.currentMa != null) maReadings.add(m.currentMa!);
        ramReadings.add(m.processRamMb);
        collectStartMah ??= m.sessionMah;
        collectEndMah = m.sessionMah;
      });

      final collectStart = DateTime.now();
      final collectEnd =
          collectStart.add(Duration(seconds: _durationSec));
      while (DateTime.now().isBefore(collectEnd) && _isRunning) {
        try {
          final sw = Stopwatch()..start();
          await yolo.predict(_syntheticBytes!, confidenceThreshold: 0.25);
          final ms = sw.elapsedMicroseconds / 1000.0;
          infTimes.add(ms);
          _pushRecentTime(ms);
        } catch (_) {/* ignora */}
      }
      await collectSub.cancel();
      final realDuration =
          DateTime.now().difference(collectStart).inMilliseconds / 1000.0;

      double? totalMah;
      if (collectStartMah != null && collectEndMah != null) {
        totalMah = collectEndMah! - collectStartMah!;
        if (totalMah < 0) totalMah = null;
      }

      _results.add(BenchmarkModelResult.from(
        modelLabel: info.label,
        modelFile: info.file,
        useGpu: useGpu,
        infTimesMs: infTimes,
        maReadings: maReadings,
        ramReadingsMb: ramReadings,
        durationSec: realDuration,
        totalMah: totalMah,
      ));
    } finally {
      _setStep(_Step.finalizing, 0);
      try {
        await yolo.dispose();
      } catch (_) {/* ignora */}
    }
  }

  void _pushRecentTime(double ms) {
    _recentInfTimes.add(ms);
    while (_recentInfTimes.length > _liveFpsWindow) {
      _recentInfTimes.removeAt(0);
    }
  }

  void _setStep(_Step step, int durationSec) {
    if (!mounted) return;
    setState(() {
      _currentStep = step;
      _stepDurationSec = durationSec;
      _stepStartTime = DateTime.now();
    });
  }

  int _stepRemaining() {
    if (_stepStartTime == null || _stepDurationSec == 0) return 0;
    final elapsed = DateTime.now().difference(_stepStartTime!).inSeconds;
    final remaining = _stepDurationSec - elapsed;
    return remaining < 0 ? 0 : remaining;
  }

  void _cancel() {
    setState(() => _isRunning = false);
  }

  // ===========================================================
  // Export JSON
  // ===========================================================
  Future<void> _exportJson() async {
    try {
      final payload = {
        'timestamp': DateTime.now().toIso8601String(),
        'duration_per_model_sec': _durationSec,
        'warmup_sec': 5,
        'stabilization_sec': 5,
        'results': _results.map((r) => r.toJson()).toList(),
      };
      final json = const JsonEncoder.withIndent('  ').convert(payload);
      final dir = await getApplicationDocumentsDirectory();
      final filename =
          'benchmark_${DateTime.now().millisecondsSinceEpoch}.json';
      final file = File('${dir.path}/$filename');
      await file.writeAsString(json);
      await Share.shareXFiles(
        [XFile(file.path)],
        text: 'Benchmark de modelos YOLO',
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Falha ao exportar: $e')),
        );
      }
    }
  }

  // ===========================================================
  // BUILD
  // ===========================================================
  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: _phase != _Phase.running,
      onPopInvokedWithResult: (didPop, _) {
        if (!didPop && _phase == _Phase.running) _cancel();
      },
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Benchmark de modelos'),
        ),
        body: switch (_phase) {
          _Phase.setup => _buildSetup(),
          _Phase.running => _buildRunning(),
          _Phase.report => _buildReport(),
        },
      ),
    );
  }

  // ----- Setup -----
  Widget _buildSetup() {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
          child: Text(
            'Selecione os modelos a comparar:',
            style: Theme.of(context).textTheme.titleMedium,
          ),
        ),
        Expanded(
          child: ListView.builder(
            itemCount: _availableModels.length,
            itemBuilder: (_, i) {
              final m = _availableModels[i];
              return CheckboxListTile(
                title: Text(m.label),
                subtitle: Text(
                  '${m.file} - ${m.useGpu ? "GPU" : "CPU"}',
                  style: const TextStyle(fontSize: 12),
                ),
                value: _selectedFiles.contains(m.file),
                onChanged: (checked) {
                  setState(() {
                    if (checked == true) {
                      _selectedFiles.add(m.file);
                    } else {
                      _selectedFiles.remove(m.file);
                    }
                  });
                },
              );
            },
          ),
        ),
        const Divider(height: 1),
        Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text('Acelerador:'),
              const SizedBox(height: 6),
              SegmentedButton<_AcceleratorMode>(
                segments: const [
                  ButtonSegment(
                    value: _AcceleratorMode.gpuOnly,
                    label: Text('GPU'),
                    icon: Icon(Icons.bolt),
                  ),
                  ButtonSegment(
                    value: _AcceleratorMode.cpuOnly,
                    label: Text('CPU'),
                    icon: Icon(Icons.memory),
                  ),
                  ButtonSegment(
                    value: _AcceleratorMode.both,
                    label: Text('Ambos'),
                    icon: Icon(Icons.compare_arrows),
                  ),
                ],
                selected: {_acceleratorMode},
                onSelectionChanged: (s) =>
                    setState(() => _acceleratorMode = s.first),
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  const Text('Duracao por modelo:'),
                  const SizedBox(width: 12),
                  DropdownButton<int>(
                    value: _durationSec,
                    items: _durationOptions
                        .map((s) => DropdownMenuItem(
                              value: s,
                              child: Text('${s}s'),
                            ))
                        .toList(),
                    onChanged: (v) {
                      if (v != null) setState(() => _durationSec = v);
                    },
                  ),
                ],
              ),
              const SizedBox(height: 4),
              Text(
                _buildEstimateText(),
                style: Theme.of(context)
                    .textTheme
                    .bodySmall
                    ?.copyWith(color: Colors.grey),
              ),
              const SizedBox(height: 12),
              SizedBox(
                height: 52,
                child: ElevatedButton.icon(
                  onPressed: _startBenchmark,
                  icon: const Icon(Icons.play_arrow),
                  label: Text(
                    _buildStartLabel(),
                    style: const TextStyle(fontSize: 16),
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  String _buildStartLabel() {
    final n = _selectedFiles.length;
    final runs = _acceleratorMode == _AcceleratorMode.both ? n * 2 : n;
    return 'Iniciar ($runs execucao${runs == 1 ? "" : "es"})';
  }

  String _buildEstimateText() {
    final n = _selectedFiles.length;
    final runs = _acceleratorMode == _AcceleratorMode.both ? n * 2 : n;
    final totalSec = runs * (10 + _durationSec);
    final minutes = (totalSec / 60).toStringAsFixed(1);
    return 'Cada execucao: 5s warmup + 5s estabilizacao + ${_durationSec}s coleta. '
        'Tempo estimado: ~$minutes min';
  }

  // ----- Running -----
  Widget _buildRunning() {
    final stepLabel = switch (_currentStep) {
      _Step.idle => 'Aguardando...',
      _Step.loading => 'Carregando modelo...',
      _Step.warmup => 'Aquecendo (warmup)...',
      _Step.stabilizing => 'Estabilizando...',
      _Step.collecting => 'Coletando metricas...',
      _Step.finalizing => 'Finalizando...',
    };
    final remaining = _stepRemaining();
    final totalProgress =
        _totalModels == 0 ? 0.0 : (_currentModelIdx) / _totalModels;
    final showCountdown = remaining > 0 &&
        (_currentStep == _Step.warmup ||
            _currentStep == _Step.stabilizing ||
            _currentStep == _Step.collecting);

    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Progresso global
          Text(
            'Modelo ${_currentModelIdx + 1} de $_totalModels',
            style: Theme.of(context).textTheme.titleSmall,
          ),
          const SizedBox(height: 4),
          LinearProgressIndicator(value: totalProgress),
          const SizedBox(height: 24),

          // Card do modelo atual
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    _currentModelLabel,
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    _currentUseGpu ? 'GPU' : 'CPU',
                    style:
                        Theme.of(context).textTheme.bodySmall?.copyWith(
                              color: Colors.grey,
                            ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      const Icon(Icons.timer, size: 20),
                      const SizedBox(width: 8),
                      Expanded(child: Text(stepLabel)),
                      if (showCountdown)
                        Text(
                          '${remaining}s',
                          style: const TextStyle(
                            fontFeatures: [FontFeature.tabularFigures()],
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Metricas ao vivo
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  Text(
                    'Metricas ao vivo',
                    style: Theme.of(context).textTheme.titleSmall,
                  ),
                  const SizedBox(height: 12),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      _liveMetric(
                        'FPS',
                        _liveFps.toStringAsFixed(1),
                      ),
                      _liveMetric(
                        'Inf ms',
                        _recentInfTimes.isEmpty
                            ? '--'
                            : (_recentInfTimes.reduce((a, b) => a + b) /
                                    _recentInfTimes.length)
                                .toStringAsFixed(1),
                      ),
                      _liveMetric(
                        'mA',
                        _liveMa?.toString() ?? '--',
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),

          if (_errorMessage != null) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.red.shade50,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                _errorMessage!,
                style: TextStyle(color: Colors.red.shade900),
              ),
            ),
          ],

          const Spacer(),
          OutlinedButton.icon(
            onPressed: _cancel,
            icon: const Icon(Icons.stop),
            label: const Text('Cancelar'),
          ),
        ],
      ),
    );
  }

  Widget _liveMetric(String label, String value) {
    return Column(
      children: [
        Text(
          label,
          style: const TextStyle(color: Colors.grey, fontSize: 12),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: const TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.w700,
            fontFeatures: [FontFeature.tabularFigures()],
          ),
        ),
      ],
    );
  }

  // ----- Report -----
  Widget _buildReport() {
    if (_results.isEmpty) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Text('Nenhum resultado coletado.'),
        ),
      );
    }

    // Identifica vencedores
    final fastestIdx = _argMin(_results.map((r) => r.avgInfMs).toList());
    final mostEfficientIdx = _argMinNullable(
      _results.map((r) => r.totalMah).toList(),
    );

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
          child: Row(
            children: [
              Expanded(
                child: Text(
                  '${_results.length} modelo${_results.length == 1 ? "" : "s"} testado${_results.length == 1 ? "" : "s"} (${_durationSec}s cada)',
                  style: Theme.of(context).textTheme.titleSmall,
                ),
              ),
              IconButton(
                icon: const Icon(Icons.share),
                tooltip: 'Compartilhar JSON',
                onPressed: _exportJson,
              ),
            ],
          ),
        ),
        Expanded(
          child: SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: SingleChildScrollView(
              child: DataTable(
                columnSpacing: 16,
                columns: const [
                  DataColumn(label: Text('Modelo')),
                  DataColumn(label: Text('GPU')),
                  DataColumn(label: Text('Avg ms'), numeric: true),
                  DataColumn(label: Text('FPS'), numeric: true),
                  DataColumn(label: Text('Avg mA'), numeric: true),
                  DataColumn(label: Text('mAh'), numeric: true),
                  DataColumn(label: Text('RAM MB'), numeric: true),
                  DataColumn(label: Text('N inf'), numeric: true),
                ],
                rows: [
                  for (var i = 0; i < _results.length; i++)
                    _buildRow(
                      _results[i],
                      isFastest: i == fastestIdx,
                      isMostEfficient: i == mostEfficientIdx,
                    ),
                ],
              ),
            ),
          ),
        ),
        const Padding(
          padding: EdgeInsets.all(12),
          child: Row(
            children: [
              SizedBox(width: 12, height: 12, child: ColoredBox(color: Color(0xFFC8E6C9))),
              SizedBox(width: 4),
              Text('mais rapido  ', style: TextStyle(fontSize: 12)),
              SizedBox(width: 12, height: 12, child: ColoredBox(color: Color(0xFFBBDEFB))),
              SizedBox(width: 4),
              Text('mais eficiente', style: TextStyle(fontSize: 12)),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          child: Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () {
                    setState(() {
                      _phase = _Phase.setup;
                      _results.clear();
                    });
                  },
                  icon: const Icon(Icons.refresh),
                  label: const Text('Novo teste'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: _exportJson,
                  icon: const Icon(Icons.share),
                  label: const Text('Compartilhar'),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  DataRow _buildRow(
    BenchmarkModelResult r, {
    required bool isFastest,
    required bool isMostEfficient,
  }) {
    Color? rowColor;
    if (isFastest && isMostEfficient) {
      rowColor = const Color(0xFFD0E8E0); // verde-azul mistura
    } else if (isFastest) {
      rowColor = const Color(0xFFC8E6C9);
    } else if (isMostEfficient) {
      rowColor = const Color(0xFFBBDEFB);
    }
    return DataRow(
      color: rowColor == null
          ? null
          : WidgetStateProperty.resolveWith((_) => rowColor),
      cells: [
        DataCell(SizedBox(
          width: 140,
          child: Text(
            r.modelLabel,
            overflow: TextOverflow.ellipsis,
          ),
        )),
        DataCell(Text(r.useGpu ? 'Sim' : 'Nao')),
        DataCell(Text(r.avgInfMs.toStringAsFixed(1))),
        DataCell(Text(r.avgFps.toStringAsFixed(1))),
        DataCell(Text(r.avgMa == null ? '--' : r.avgMa!.toStringAsFixed(0))),
        DataCell(Text(
          r.totalMah == null ? '--' : r.totalMah!.toStringAsFixed(2),
        )),
        DataCell(Text(r.avgRamMb.toStringAsFixed(0))),
        DataCell(Text('${r.totalInferences}')),
      ],
    );
  }

  /// Retorna o indice do menor valor na lista. -1 se vazia.
  int _argMin(List<double> values) {
    if (values.isEmpty) return -1;
    var idx = 0;
    var min = values[0];
    for (var i = 1; i < values.length; i++) {
      if (values[i] < min) {
        min = values[i];
        idx = i;
      }
    }
    return idx;
  }

  /// Argmin ignorando nulls. -1 se nenhum valor valido.
  int _argMinNullable(List<double?> values) {
    var idx = -1;
    double? min;
    for (var i = 0; i < values.length; i++) {
      final v = values[i];
      if (v == null) continue;
      if (min == null || v < min) {
        min = v;
        idx = i;
      }
    }
    return idx;
  }
}
