/// Tipo da regra de infracao.
///
/// - [presence]: classe na lista, se detectada acima do threshold,
///   dispara infracao (ex: 'NO-Hardhat').
/// - [absence]: classe deve estar presente; se nao for vista por mais
///   de [InfractionRuleSet.absenceWindowMs], dispara infracao
///   (ex: ausencia de 'Hardhat').
enum RuleKind { presence, absence }

class InfractionRule {
  final String classLabel;
  final RuleKind kind;

  const InfractionRule({
    required this.classLabel,
    required this.kind,
  });

  Map<String, dynamic> toJson() => {
        'classLabel': classLabel,
        'kind': kind.name,
      };

  factory InfractionRule.fromJson(Map<String, dynamic> json) => InfractionRule(
        classLabel: json['classLabel'] as String,
        kind: RuleKind.values.firstWhere(
          (k) => k.name == json['kind'],
          orElse: () => RuleKind.presence,
        ),
      );
}

/// Conjunto de regras associadas a um modelo.
///
/// `modelKey` identifica de forma estavel o modelo (filename para bundled,
/// path absoluto para custom). `absenceWindowMs` controla a janela
/// deslizante usada na avaliacao de regras de ausencia.
class InfractionRuleSet {
  final String modelKey;
  final List<InfractionRule> rules;
  final int absenceWindowMs;
  final bool requirePersonForAbsence;

  const InfractionRuleSet({
    required this.modelKey,
    required this.rules,
    this.absenceWindowMs = 2500,
    this.requirePersonForAbsence = true,
  });

  const InfractionRuleSet.empty(this.modelKey)
      : rules = const [],
        absenceWindowMs = 2500,
        requirePersonForAbsence = true;

  bool get isEmpty => rules.isEmpty;

  Iterable<String> get presenceClasses =>
      rules.where((r) => r.kind == RuleKind.presence).map((r) => r.classLabel);

  Iterable<String> get absenceClasses =>
      rules.where((r) => r.kind == RuleKind.absence).map((r) => r.classLabel);

  InfractionRuleSet copyWith({
    List<InfractionRule>? rules,
    int? absenceWindowMs,
    bool? requirePersonForAbsence,
  }) {
    return InfractionRuleSet(
      modelKey: modelKey,
      rules: rules ?? this.rules,
      absenceWindowMs: absenceWindowMs ?? this.absenceWindowMs,
      requirePersonForAbsence:
          requirePersonForAbsence ?? this.requirePersonForAbsence,
    );
  }

  Map<String, dynamic> toJson() => {
        'modelKey': modelKey,
        'rules': rules.map((r) => r.toJson()).toList(),
        'absenceWindowMs': absenceWindowMs,
        'requirePersonForAbsence': requirePersonForAbsence,
      };

  factory InfractionRuleSet.fromJson(Map<String, dynamic> json) {
    final rawRules = json['rules'];
    return InfractionRuleSet(
      modelKey: json['modelKey'] as String,
      rules: rawRules is List
          ? rawRules
              .whereType<Map>()
              .map((m) => InfractionRule.fromJson(Map<String, dynamic>.from(m)))
              .toList()
          : const [],
      absenceWindowMs: (json['absenceWindowMs'] as num?)?.toInt() ?? 2500,
      requirePersonForAbsence: json['requirePersonForAbsence'] as bool? ?? true,
    );
  }

  /// Constroi um [InfractionRuleSet] a partir do bloco
  /// `defaultInfractionRules` do `models.json`.
  factory InfractionRuleSet.fromManifest(
    String modelKey,
    Map<String, dynamic>? raw,
  ) {
    if (raw == null) return InfractionRuleSet.empty(modelKey);
    final presence = (raw['presence'] as List?)?.whereType<String>() ?? const [];
    final absence = (raw['absence'] as List?)?.whereType<String>() ?? const [];
    final rules = <InfractionRule>[
      for (final c in presence)
        InfractionRule(classLabel: c, kind: RuleKind.presence),
      for (final c in absence)
        InfractionRule(classLabel: c, kind: RuleKind.absence),
    ];
    return InfractionRuleSet(
      modelKey: modelKey,
      rules: rules,
      absenceWindowMs: (raw['absence_window_ms'] as num?)?.toInt() ?? 2500,
      requirePersonForAbsence:
          raw['require_person_for_absence'] as bool? ?? true,
    );
  }
}
