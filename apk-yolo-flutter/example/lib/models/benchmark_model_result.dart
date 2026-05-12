/// Resultado consolidado do benchmark de um unico modelo.
///
/// Calculado a partir das listas brutas coletadas durante a fase de
/// medicao em `BenchmarkScreen`. As listas em si nao sao persistidas --
/// apenas as estatisticas agregadas.
class BenchmarkModelResult {
  final String modelLabel;
  final String modelFile;
  final bool useGpu;

  final int totalInferences;
  final double avgInfMs;
  final double minInfMs;
  final double maxInfMs;

  /// FPS efetivo (totalInferences / durationSec).
  final double avgFps;

  /// Media de corrente instantanea (mA) durante a coleta.
  /// `null` quando nenhuma leitura foi obtida (ex: iOS, dispositivo sem suporte).
  final double? avgMa;

  /// mAh consumido durante a fase de coleta deste modelo.
  /// `null` quando nao ha leituras de corrente disponiveis.
  final double? totalMah;

  /// Memoria residente media do processo Flutter (MB) durante a coleta.
  final double avgRamMb;

  /// Duracao real da fase de coleta (pode diferir do alvo se o usuario
  /// cancelou ou se houve atraso na finalizacao do loop).
  final double durationSec;

  const BenchmarkModelResult({
    required this.modelLabel,
    required this.modelFile,
    required this.useGpu,
    required this.totalInferences,
    required this.avgInfMs,
    required this.minInfMs,
    required this.maxInfMs,
    required this.avgFps,
    required this.avgMa,
    required this.totalMah,
    required this.avgRamMb,
    required this.durationSec,
  });

  /// Constroi a partir das listas brutas coletadas.
  factory BenchmarkModelResult.from({
    required String modelLabel,
    required String modelFile,
    required bool useGpu,
    required List<double> infTimesMs,
    required List<int> maReadings,
    required List<double> ramReadingsMb,
    required double durationSec,
    required double? totalMah,
  }) {
    final n = infTimesMs.length;
    final avgInf = n == 0
        ? 0.0
        : infTimesMs.reduce((a, b) => a + b) / n;
    final minInf = n == 0
        ? 0.0
        : infTimesMs.reduce((a, b) => a < b ? a : b);
    final maxInf = n == 0
        ? 0.0
        : infTimesMs.reduce((a, b) => a > b ? a : b);
    final fps = durationSec <= 0 ? 0.0 : n / durationSec;
    final avgMa = maReadings.isEmpty
        ? null
        : maReadings.reduce((a, b) => a + b) / maReadings.length;
    final avgRam = ramReadingsMb.isEmpty
        ? 0.0
        : ramReadingsMb.reduce((a, b) => a + b) / ramReadingsMb.length;

    return BenchmarkModelResult(
      modelLabel: modelLabel,
      modelFile: modelFile,
      useGpu: useGpu,
      totalInferences: n,
      avgInfMs: avgInf,
      minInfMs: minInf,
      maxInfMs: maxInf,
      avgFps: fps,
      avgMa: avgMa,
      totalMah: totalMah,
      avgRamMb: avgRam,
      durationSec: durationSec,
    );
  }

  Map<String, dynamic> toJson() => {
        'model_label': modelLabel,
        'model_file': modelFile,
        'use_gpu': useGpu,
        'total_inferences': totalInferences,
        'avg_inf_ms': _round(avgInfMs, 3),
        'min_inf_ms': _round(minInfMs, 3),
        'max_inf_ms': _round(maxInfMs, 3),
        'avg_fps': _round(avgFps, 2),
        'avg_ma': avgMa == null ? null : _round(avgMa!, 1),
        'total_mah': totalMah == null ? null : _round(totalMah!, 3),
        'avg_ram_mb': _round(avgRamMb, 1),
        'duration_sec': _round(durationSec, 2),
      };

  static double _round(double v, int decimals) {
    final factor = _pow10(decimals);
    return (v * factor).round() / factor;
  }

  static double _pow10(int n) {
    var r = 1.0;
    for (var i = 0; i < n; i++) {
      r *= 10;
    }
    return r;
  }
}
