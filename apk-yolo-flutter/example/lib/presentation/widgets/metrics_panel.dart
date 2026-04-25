import 'package:flutter/material.dart';

import '../../services/system_metrics_service.dart';

/// Painel colapsavel no canto superior direito que exibe FPS,
/// tempo de inferencia, RAM e bateria.
///
/// Estado expandido: tabela completa. Estado colapsado: pilula
/// compacta `FPS x.x · y%`.
class MetricsPanel extends StatefulWidget {
  const MetricsPanel({
    super.key,
    required this.fps,
    required this.processingTimeMs,
    required this.detectionCount,
    required this.metrics,
  });

  final double fps;
  final double processingTimeMs;
  final int detectionCount;
  final SystemMetrics? metrics;

  @override
  State<MetricsPanel> createState() => _MetricsPanelState();
}

class _MetricsPanelState extends State<MetricsPanel> {
  bool _expanded = true;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.black.withValues(alpha: 0.55),
      borderRadius: BorderRadius.circular(10),
      child: InkWell(
        borderRadius: BorderRadius.circular(10),
        onTap: () => setState(() => _expanded = !_expanded),
        child: AnimatedSize(
          duration: const Duration(milliseconds: 180),
          curve: Curves.easeOut,
          alignment: Alignment.topRight,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
            child: _expanded ? _buildExpanded() : _buildCollapsed(),
          ),
        ),
      ),
    );
  }

  Widget _buildCollapsed() {
    final battery = widget.metrics?.batteryPercent;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          'FPS ${widget.fps.toStringAsFixed(1)}',
          style: _valueStyle,
        ),
        if (battery != null) ...[
          const SizedBox(width: 8),
          Text('· $battery%', style: _valueStyle),
        ],
        const SizedBox(width: 4),
        const Icon(Icons.expand_more, color: Colors.white, size: 16),
      ],
    );
  }

  Widget _buildExpanded() {
    final m = widget.metrics;
    final ramText = '${widget.metrics?.processRamMb.toStringAsFixed(0) ?? '--'} MB';
    final totalRam = m?.deviceTotalRamMb;
    final ramFull = totalRam != null
        ? '$ramText / ${totalRam.toStringAsFixed(0)} MB'
        : ramText;
    final batteryText = m?.batteryPercent != null ? '${m!.batteryPercent}%' : '--';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('Metricas',
                style: TextStyle(
                  color: Colors.white70,
                  fontSize: 11,
                  letterSpacing: 0.5,
                  fontWeight: FontWeight.w600,
                )),
            SizedBox(width: 6),
            Icon(Icons.expand_less, color: Colors.white, size: 16),
          ],
        ),
        const SizedBox(height: 4),
        _metricRow('FPS', widget.fps.toStringAsFixed(1)),
        _metricRow('Inf', '${widget.processingTimeMs.toStringAsFixed(1)} ms'),
        _metricRow('Det', '${widget.detectionCount}'),
        _metricRow('RAM', ramFull),
        _metricRow('Bat', batteryText),
      ],
    );
  }

  Widget _metricRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 1),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 32,
            child: Text(label, style: _labelStyle),
          ),
          Text(value, style: _valueStyle),
        ],
      ),
    );
  }

  static const _labelStyle = TextStyle(
    color: Colors.white60,
    fontSize: 11,
    fontWeight: FontWeight.w500,
  );
  static const _valueStyle = TextStyle(
    color: Colors.white,
    fontSize: 12,
    fontWeight: FontWeight.w600,
    fontFeatures: [FontFeature.tabularFigures()],
  );
}
