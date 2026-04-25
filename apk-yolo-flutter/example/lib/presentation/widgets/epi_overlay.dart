import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

/// Overlay customizado que pinta as bboxes detectadas em duas cores:
/// vermelho para detec coes em [redBoxes] (infracoes) e verde para
/// as demais. Substitui o overlay default do plugin (que pinta tudo
/// na mesma cor).
class EpiOverlay extends StatelessWidget {
  const EpiOverlay({
    super.key,
    required this.detections,
    required this.redBoxes,
  });

  final List<YOLOResult> detections;
  final List<YOLOResult> redBoxes;

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: CustomPaint(
        painter: _EpiBoxPainter(
          detections: detections,
          redKeys: redBoxes.map(_keyOf).toSet(),
        ),
        size: Size.infinite,
      ),
    );
  }

  static String _keyOf(YOLOResult r) =>
      '${r.className}_${r.normalizedBox.left.toStringAsFixed(4)}_${r.normalizedBox.top.toStringAsFixed(4)}';
}

class _EpiBoxPainter extends CustomPainter {
  _EpiBoxPainter({required this.detections, required this.redKeys});

  final List<YOLOResult> detections;
  final Set<String> redKeys;

  static const _greenStroke = Color(0xFF4CAF50);
  static const _redStroke = Color(0xFFE53935);
  static const _greenFill = Color(0x334CAF50);
  static const _redFill = Color(0x66E53935);

  @override
  void paint(Canvas canvas, Size size) {
    if (detections.isEmpty) return;

    final strokePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    final fillPaint = Paint()..style = PaintingStyle.fill;

    for (final d in detections) {
      final isRed = redKeys.contains(EpiOverlay._keyOf(d));
      final box = d.normalizedBox;
      final rect = Rect.fromLTRB(
        box.left * size.width,
        box.top * size.height,
        box.right * size.width,
        box.bottom * size.height,
      );

      strokePaint.color = isRed ? _redStroke : _greenStroke;
      fillPaint.color = isRed ? _redFill : _greenFill;

      canvas.drawRect(rect, fillPaint);
      canvas.drawRect(rect, strokePaint);

      _drawLabel(
        canvas,
        rect,
        '${d.className} ${(d.confidence * 100).toStringAsFixed(0)}%',
        isRed ? _redStroke : _greenStroke,
      );
    }
  }

  void _drawLabel(Canvas canvas, Rect box, String text, Color bg) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 11,
          fontWeight: FontWeight.w600,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    const padding = 4.0;
    final labelRect = Rect.fromLTWH(
      box.left,
      (box.top - tp.height - padding * 2).clamp(0.0, double.infinity),
      tp.width + padding * 2,
      tp.height + padding * 2,
    );
    canvas.drawRect(labelRect, Paint()..color = bg);
    tp.paint(canvas, Offset(labelRect.left + padding, labelRect.top + padding));
  }

  @override
  bool shouldRepaint(_EpiBoxPainter old) =>
      old.detections != detections || old.redKeys != redKeys;
}
