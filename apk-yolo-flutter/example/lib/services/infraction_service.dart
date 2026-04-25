import 'package:ultralytics_yolo/models/yolo_result.dart';

import '../models/infraction_rule.dart';

/// Resultado da avaliacao de regras de infracao em um frame.
class InfractionEvaluation {
  /// `true` quando ha pelo menos uma regra disparada.
  final bool hasInfraction;

  /// Classes que dispararam infracao (nomes legiveis).
  final List<String> offendingClasses;

  /// Bboxes que devem ser pintadas em vermelho na overlay.
  /// Subconjunto das deteccoes do frame correspondentes a regras
  /// `presence`. Regras `absence` nao tem bbox associada.
  final List<YOLOResult> redBoxes;

  const InfractionEvaluation({
    required this.hasInfraction,
    required this.offendingClasses,
    required this.redBoxes,
  });

  static const empty = InfractionEvaluation(
    hasInfraction: false,
    offendingClasses: [],
    redBoxes: [],
  );
}

/// Avalia regras de infracao em cada frame de deteccao.
///
/// Mantem uma janela deslizante por classe para regras `absence`:
/// uma classe so e considerada "ausente" se nao aparecer em nenhum
/// frame durante `absenceWindowMs`. Isto evita alarmes piscando
/// quando ha pequenas falhas de deteccao entre frames.
class InfractionService {
  InfractionService();

  /// Ultima vez (em ms desde epoch) que cada classe foi vista.
  final Map<String, int> _lastSeenMs = {};

  /// Buffers reusados a cada frame para evitar alocacoes.
  final Set<String> _seenInFrame = {};

  /// Marca todas as classes esperadas para que a janela de ausencia
  /// comece a contar a partir de agora -- evita disparar infracao no
  /// primeiro frame apos trocar de modelo.
  void primeWindow(Iterable<String> expectedClasses) {
    final now = DateTime.now().millisecondsSinceEpoch;
    for (final c in expectedClasses) {
      _lastSeenMs[c] = now;
    }
  }

  void resetWindow() {
    _lastSeenMs.clear();
  }

  /// Avalia [detections] contra [rules]. [modelClasses] descreve as
  /// classes que o modelo pode emitir -- se vazia ou nao incluir a
  /// classe da regra, a regra e ignorada (modo seguro).
  InfractionEvaluation evaluate(
    List<YOLOResult> detections,
    InfractionRuleSet rules, {
    required Set<String> modelClasses,
  }) {
    if (rules.isEmpty) return InfractionEvaluation.empty;

    final now = DateTime.now().millisecondsSinceEpoch;
    _seenInFrame.clear();
    for (final d in detections) {
      _seenInFrame.add(d.className);
      _lastSeenMs[d.className] = now;
    }

    final offending = <String>{};
    final redBoxes = <YOLOResult>[];

    // Regras presence: classe na lista detectada => infracao.
    final presence = rules.presenceClasses.toSet();
    if (presence.isNotEmpty) {
      for (final d in detections) {
        if (presence.contains(d.className)) {
          offending.add(d.className);
          redBoxes.add(d);
        }
      }
    }

    // Regras absence: classe esperada nao vista na janela => infracao.
    // Ignora se require_person_for_absence e nao ha pessoas no frame.
    final absence = rules.absenceClasses.toSet();
    if (absence.isNotEmpty) {
      final hasPersonInFrame = _seenInFrame.contains('Person');
      if (!rules.requirePersonForAbsence || hasPersonInFrame) {
        final cutoff = now - rules.absenceWindowMs;
        for (final cls in absence) {
          // Ignora classes que o modelo nem emite.
          if (modelClasses.isNotEmpty && !modelClasses.contains(cls)) continue;
          // Ignora se conflitar com presence (presence vence).
          if (presence.contains(cls)) continue;
          final lastSeen = _lastSeenMs[cls];
          if (lastSeen == null || lastSeen < cutoff) {
            offending.add('Falta: $cls');
          }
        }
      }
    }

    final has = offending.isNotEmpty;
    return InfractionEvaluation(
      hasInfraction: has,
      offendingClasses: offending.toList(growable: false),
      redBoxes: redBoxes,
    );
  }
}
