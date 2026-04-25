import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/infraction_rule.dart';

/// Persiste overrides de [InfractionRuleSet] por modelo em
/// SharedPreferences. Cada modelo tem uma chave isolada
/// (`epi_rules_<modelKey>`) para que regras nao vazem entre modelos.
///
/// O storage e in-memory + SharedPreferences -- as leituras apos o
/// [load] inicial sao sincronas (`ruleSetFor`).
class InfractionRulesStorage extends ChangeNotifier {
  InfractionRulesStorage._(this._prefs);

  static const _keyPrefix = 'epi_rules_';

  static InfractionRulesStorage? _instance;

  static InfractionRulesStorage get instance {
    final i = _instance;
    if (i == null) {
      throw StateError(
        'InfractionRulesStorage.load() must be called before access.',
      );
    }
    return i;
  }

  final SharedPreferences _prefs;
  final Map<String, InfractionRuleSet> _cache = {};

  static Future<void> load() async {
    if (_instance != null) return;
    final prefs = await SharedPreferences.getInstance();
    _instance = InfractionRulesStorage._(prefs);
  }

  /// Retorna o ruleset ativo para [modelKey].
  /// - Se houver override persistido, retorna ele.
  /// - Caso contrario retorna [defaults].
  InfractionRuleSet ruleSetFor(
    String modelKey,
    InfractionRuleSet defaults,
  ) {
    final cached = _cache[modelKey];
    if (cached != null) return cached;

    final raw = _prefs.getString(_storageKey(modelKey));
    if (raw == null) {
      _cache[modelKey] = defaults;
      return defaults;
    }
    try {
      final json = jsonDecode(raw) as Map<String, dynamic>;
      final rs = InfractionRuleSet.fromJson(json);
      _cache[modelKey] = rs;
      return rs;
    } catch (e) {
      debugPrint('InfractionRulesStorage: invalid json for $modelKey ($e)');
      _cache[modelKey] = defaults;
      return defaults;
    }
  }

  /// Salva [rs] como override do modelo [rs.modelKey].
  Future<void> override(InfractionRuleSet rs) async {
    _cache[rs.modelKey] = rs;
    await _prefs.setString(
      _storageKey(rs.modelKey),
      jsonEncode(rs.toJson()),
    );
    notifyListeners();
  }

  /// Remove override e volta para defaults na proxima leitura.
  Future<void> reset(String modelKey) async {
    _cache.remove(modelKey);
    await _prefs.remove(_storageKey(modelKey));
    notifyListeners();
  }

  String _storageKey(String modelKey) => '$_keyPrefix$modelKey';
}
