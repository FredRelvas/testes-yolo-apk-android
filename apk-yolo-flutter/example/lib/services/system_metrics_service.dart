import 'dart:async';
import 'dart:io';

import 'package:battery_plus/battery_plus.dart';
import 'package:flutter/foundation.dart';
import 'package:system_info_plus/system_info_plus.dart';

/// Snapshot periodico de metricas do sistema (RAM + bateria).
///
/// `processRamMb` mede a memoria residente do processo Flutter via
/// `dart:io`. `deviceTotalRamMb` e fixo por dispositivo. `batteryPercent`
/// e amostrado via battery_plus a cada 1s.
class SystemMetrics {
  final double processRamMb;
  final double? deviceTotalRamMb;
  final int? batteryPercent;
  final BatteryState? batteryState;
  final DateTime timestamp;

  const SystemMetrics({
    required this.processRamMb,
    required this.deviceTotalRamMb,
    required this.batteryPercent,
    required this.batteryState,
    required this.timestamp,
  });

  static SystemMetrics zero() => SystemMetrics(
        processRamMb: 0,
        deviceTotalRamMb: null,
        batteryPercent: null,
        batteryState: null,
        timestamp: DateTime.now(),
      );
}

/// Stream de [SystemMetrics] a 1Hz. Use [start] uma unica vez na app
/// e ouca [stream] de quem quiser exibir as metricas.
class SystemMetricsService {
  SystemMetricsService._();

  static final SystemMetricsService instance = SystemMetricsService._();

  final _battery = Battery();
  final _controller = StreamController<SystemMetrics>.broadcast();
  Timer? _timer;
  double? _deviceTotalRamMb;

  Stream<SystemMetrics> get stream => _controller.stream;

  Future<void> start({Duration interval = const Duration(seconds: 1)}) async {
    if (_timer != null) return;
    _deviceTotalRamMb = await _readDeviceTotalRamMb();
    _timer = Timer.periodic(interval, (_) => _tick());
    // Primeiro tick imediato para nao esperar 1s.
    _tick();
  }

  void stop() {
    _timer?.cancel();
    _timer = null;
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

      _controller.add(SystemMetrics(
        processRamMb: processMb,
        deviceTotalRamMb: _deviceTotalRamMb,
        batteryPercent: batteryLevel,
        batteryState: batteryState,
        timestamp: DateTime.now(),
      ));
    } catch (e) {
      debugPrint('SystemMetricsService: tick error ($e)');
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
