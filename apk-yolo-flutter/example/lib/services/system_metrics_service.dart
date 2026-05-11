import 'dart:async';
import 'dart:collection';
import 'dart:io';

import 'package:battery_plus/battery_plus.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:system_info_plus/system_info_plus.dart';

/// Snapshot periodico de metricas do sistema (RAM + bateria + corrente).
///
/// - `processRamMb`: memoria residente do processo Flutter (via dart:io)
/// - `deviceTotalRamMb`: RAM fisica do dispositivo
/// - `batteryPercent` / `batteryState`: amostrados via battery_plus
/// - `currentMa`: corrente instantanea (BatteryManager.CURRENT_NOW)
/// - `avgMaWindow`: media movel das ultimas N amostras de [currentMa]
/// - `sessionMah`: integral de [currentMa] desde [SystemMetricsService.start]
/// - `sessionDurationSec`: tempo desde o start
class SystemMetrics {
  final double processRamMb;
  final double? deviceTotalRamMb;
  final int? batteryPercent;
  final BatteryState? batteryState;
  final int? currentMa;
  final double? avgMaWindow;
  final double sessionMah;
  final int sessionDurationSec;
  final DateTime timestamp;

  const SystemMetrics({
    required this.processRamMb,
    required this.deviceTotalRamMb,
    required this.batteryPercent,
    required this.batteryState,
    required this.currentMa,
    required this.avgMaWindow,
    required this.sessionMah,
    required this.sessionDurationSec,
    required this.timestamp,
  });

  static SystemMetrics zero() => SystemMetrics(
        processRamMb: 0,
        deviceTotalRamMb: null,
        batteryPercent: null,
        batteryState: null,
        currentMa: null,
        avgMaWindow: null,
        sessionMah: 0,
        sessionDurationSec: 0,
        timestamp: DateTime.now(),
      );
}

/// Stream de [SystemMetrics] a 1Hz. Use [start] uma unica vez na app
/// e ouca [stream] de quem quiser exibir as metricas.
class SystemMetricsService {
  SystemMetricsService._();

  static final SystemMetricsService instance = SystemMetricsService._();

  static const _powerChannel = MethodChannel('poc_epi/power');

  /// Tamanho da janela usada para [SystemMetrics.avgMaWindow].
  static const int _avgWindowSamples = 30;

  final _battery = Battery();
  final _controller = StreamController<SystemMetrics>.broadcast();
  final Queue<int> _maWindow = Queue<int>();

  Timer? _timer;
  double? _deviceTotalRamMb;
  DateTime? _sessionStart;
  double _sessionMah = 0;
  Duration _tickInterval = const Duration(seconds: 1);

  Stream<SystemMetrics> get stream => _controller.stream;

  Future<void> start({Duration interval = const Duration(seconds: 1)}) async {
    if (_timer != null) return;
    _tickInterval = interval;
    _sessionStart = DateTime.now();
    _sessionMah = 0;
    _maWindow.clear();
    _deviceTotalRamMb = await _readDeviceTotalRamMb();
    _timer = Timer.periodic(interval, (_) => _tick());
    _tick();
  }

  void stop() {
    _timer?.cancel();
    _timer = null;
  }

  /// Zera o acumulado da sessao (`sessionMah` + duracao).
  /// Util para um botao "resetar contador" na UI.
  void resetSession() {
    _sessionStart = DateTime.now();
    _sessionMah = 0;
    _maWindow.clear();
  }

  Future<void> _tick() async {
    if (_controller.isClosed) return;
    try {
      final processRss = ProcessInfo.currentRss;
      final processMb = processRss / (1024 * 1024);

      int? batteryLevel;
      BatteryState? batteryState;
      try {
        batteryLevel = await _battery.batteryLevel;
        batteryState = await _battery.batteryState;
      } catch (e) {
        debugPrint('SystemMetricsService: battery read failed ($e)');
      }

      final currentMa = await _readCurrentMa();
      if (currentMa != null) {
        _maWindow.addLast(currentMa);
        while (_maWindow.length > _avgWindowSamples) {
          _maWindow.removeFirst();
        }
        // Integra: mA * (intervalo em h) -> mAh consumido neste tick.
        final hours = _tickInterval.inMilliseconds / 3600000.0;
        _sessionMah += currentMa * hours;
      }

      final avg = _maWindow.isEmpty
          ? null
          : _maWindow.fold<int>(0, (a, b) => a + b) / _maWindow.length;
      final sessionDurationSec = _sessionStart == null
          ? 0
          : DateTime.now().difference(_sessionStart!).inSeconds;

      _controller.add(SystemMetrics(
        processRamMb: processMb,
        deviceTotalRamMb: _deviceTotalRamMb,
        batteryPercent: batteryLevel,
        batteryState: batteryState,
        currentMa: currentMa,
        avgMaWindow: avg,
        sessionMah: _sessionMah,
        sessionDurationSec: sessionDurationSec,
        timestamp: DateTime.now(),
      ));
    } catch (e) {
      debugPrint('SystemMetricsService: tick error ($e)');
    }
  }

  /// Le corrente instantanea do MethodChannel nativo (microamperes)
  /// e converte para mA (magnitude). Retorna `null` se nao suportado
  /// ou se o canal nao estiver disponivel (ex: iOS).
  Future<int?> _readCurrentMa() async {
    if (!Platform.isAndroid) return null;
    try {
      final ua = await _powerChannel.invokeMethod<num?>('currentNowMicroAmps');
      if (ua == null) return null;
      return (ua.abs() / 1000).round();
    } on MissingPluginException {
      return null;
    } catch (e) {
      debugPrint('SystemMetricsService: currentMa read failed ($e)');
      return null;
    }
  }

  Future<double?> _readDeviceTotalRamMb() async {
    try {
      final bytes = await SystemInfoPlus.physicalMemory;
      if (bytes == null) return null;
      return bytes / (1024 * 1024);
    } catch (e) {
      debugPrint('SystemMetricsService: device ram read failed ($e)');
      return null;
    }
  }
}
