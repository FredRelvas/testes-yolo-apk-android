import 'package:flutter/material.dart';

/// Banner vermelho pulsante exibido no topo da camera quando ha
/// infracao detectada. Mostra a lista de classes ofensivas.
///
/// Quando [classes] estiver vazia, o banner se esconde com fade.
class InfractionAlertBanner extends StatefulWidget {
  const InfractionAlertBanner({super.key, required this.classes});

  final List<String> classes;

  @override
  State<InfractionAlertBanner> createState() => _InfractionAlertBannerState();
}

class _InfractionAlertBannerState extends State<InfractionAlertBanner>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulse;

  @override
  void initState() {
    super.initState();
    _pulse = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final visible = widget.classes.isNotEmpty;
    return AnimatedOpacity(
      opacity: visible ? 1.0 : 0.0,
      duration: const Duration(milliseconds: 200),
      child: AnimatedBuilder(
        animation: _pulse,
        builder: (context, child) {
          final t = _pulse.value;
          return Container(
            decoration: BoxDecoration(
              color: Color.lerp(
                const Color(0xFFB71C1C),
                const Color(0xFFE53935),
                t,
              ),
              borderRadius: BorderRadius.circular(12),
              boxShadow: [
                BoxShadow(
                  color: Colors.red.withValues(alpha: 0.3 + 0.2 * t),
                  blurRadius: 12 + 6 * t,
                  spreadRadius: 1,
                ),
              ],
            ),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            child: child,
          );
        },
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.warning_amber_rounded,
                color: Colors.white, size: 22),
            const SizedBox(width: 8),
            Flexible(
              child: Text(
                visible
                    ? 'INFRACAO: ${widget.classes.join(', ')}'
                    : 'OK',
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 14,
                  letterSpacing: 0.5,
                ),
                overflow: TextOverflow.ellipsis,
                maxLines: 2,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
