import 'package:flutter/material.dart';

import '../../models/infraction_rule.dart';
import '../../models/model_descriptor.dart';

/// Modal bottom sheet de configuracao das regras de infracao para
/// um modelo. Permite marcar cada classe como:
///   - presence (detectada gera infracao),
///   - absence (ausente gera infracao),
///   - none (ignorada).
///
/// E ajustar a janela deslizante (`absence_window_ms`) e a flag de
/// "exigir pessoa para ausencia".
///
/// Retorna o novo [InfractionRuleSet] em `Navigator.pop`, ou `null`
/// se o usuario cancelar. Botao "Restaurar padrao" retorna o
/// [model.defaultRules] (e ainda precisa ser persistido pelo caller).
class InfractionConfigSheet extends StatefulWidget {
  const InfractionConfigSheet({
    super.key,
    required this.model,
    required this.current,
  });

  final ModelDescriptor model;
  final InfractionRuleSet current;

  @override
  State<InfractionConfigSheet> createState() => _InfractionConfigSheetState();
}

class _InfractionConfigSheetState extends State<InfractionConfigSheet> {
  late Map<String, RuleKind?> _selection;
  late int _windowMs;
  late bool _requirePerson;
  late TextEditingController _customClassesCtrl;

  @override
  void initState() {
    super.initState();
    final classes = widget.model.classes ?? const <String>[];
    _selection = {for (final c in classes) c: null};
    for (final r in widget.current.rules) {
      _selection[r.classLabel] = r.kind;
    }
    _windowMs = widget.current.absenceWindowMs;
    _requirePerson = widget.current.requirePersonForAbsence;
    _customClassesCtrl = TextEditingController(
      text: classes.isEmpty
          ? widget.current.rules.map((r) => r.classLabel).join('\n')
          : '',
    );
  }

  @override
  void dispose() {
    _customClassesCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final hasClasses = (widget.model.classes ?? const []).isNotEmpty;

    return SafeArea(
      child: DraggableScrollableSheet(
        initialChildSize: 0.85,
        minChildSize: 0.5,
        maxChildSize: 0.95,
        expand: false,
        builder: (context, scrollController) {
          return Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                _buildHandle(),
                _buildHeader(),
                const SizedBox(height: 8),
                Expanded(
                  child: ListView(
                    controller: scrollController,
                    children: [
                      if (hasClasses)
                        _buildClassList()
                      else
                        _buildCustomClassesEditor(),
                      const Divider(height: 32),
                      _buildWindowSlider(),
                      _buildPersonGuard(),
                    ],
                  ),
                ),
                _buildButtons(),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildHandle() => Container(
        width: 40,
        height: 4,
        margin: const EdgeInsets.only(bottom: 12),
        decoration: BoxDecoration(
          color: Colors.grey,
          borderRadius: BorderRadius.circular(2),
        ),
      );

  Widget _buildHeader() => Align(
        alignment: Alignment.centerLeft,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Regras de infracao',
              style: Theme.of(context)
                  .textTheme
                  .titleLarge
                  ?.copyWith(fontWeight: FontWeight.bold),
            ),
            Text(
              'Modelo: ${widget.model.label}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      );

  Widget _buildClassList() {
    final classes = widget.model.classes ?? const <String>[];
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const _RulesLegend(),
        const SizedBox(height: 8),
        for (final cls in classes)
          _ClassRuleTile(
            classLabel: cls,
            current: _selection[cls],
            onChanged: (k) => setState(() => _selection[cls] = k),
          ),
      ],
    );
  }

  Widget _buildCustomClassesEditor() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Modelo sem classes declaradas. Liste as classes que '
          'disparam infracao (uma por linha):',
          style: TextStyle(fontSize: 12, color: Colors.black54),
        ),
        const SizedBox(height: 8),
        TextField(
          controller: _customClassesCtrl,
          maxLines: 6,
          decoration: const InputDecoration(
            hintText: 'NO-Hardhat\nNO-Mask\n...',
            border: OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: 4),
        const Text(
          'Todas as classes acima viram regras "presence" '
          '(detectada => infracao).',
          style: TextStyle(fontSize: 11, color: Colors.black45),
        ),
      ],
    );
  }

  Widget _buildWindowSlider() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Janela para regras de ausencia: $_windowMs ms',
          style: Theme.of(context).textTheme.bodyMedium,
        ),
        Slider(
          value: _windowMs.toDouble(),
          min: 1000,
          max: 5000,
          divisions: 8,
          label: '$_windowMs ms',
          onChanged: (v) => setState(() => _windowMs = v.toInt()),
        ),
      ],
    );
  }

  Widget _buildPersonGuard() {
    return SwitchListTile(
      contentPadding: EdgeInsets.zero,
      value: _requirePerson,
      onChanged: (v) => setState(() => _requirePerson = v),
      title: const Text('Exigir pessoa em cena para regras de ausencia'),
      subtitle: const Text(
        'Evita falso-positivo quando ninguem esta na imagem.',
        style: TextStyle(fontSize: 11),
      ),
    );
  }

  Widget _buildButtons() {
    return Row(
      children: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancelar'),
        ),
        const Spacer(),
        TextButton(
          onPressed: _restoreDefaults,
          child: const Text('Restaurar padrao'),
        ),
        const SizedBox(width: 8),
        FilledButton(
          onPressed: _apply,
          child: const Text('Aplicar'),
        ),
      ],
    );
  }

  void _restoreDefaults() {
    Navigator.pop(context, widget.model.defaultRules);
  }

  void _apply() {
    final rules = <InfractionRule>[];
    final hasClasses = (widget.model.classes ?? const []).isNotEmpty;
    if (hasClasses) {
      _selection.forEach((cls, kind) {
        if (kind != null) {
          rules.add(InfractionRule(classLabel: cls, kind: kind));
        }
      });
    } else {
      final lines = _customClassesCtrl.text
          .split(RegExp(r'\r?\n'))
          .map((l) => l.trim())
          .where((l) => l.isNotEmpty);
      for (final cls in lines) {
        rules.add(InfractionRule(classLabel: cls, kind: RuleKind.presence));
      }
    }
    final newRules = InfractionRuleSet(
      modelKey: widget.model.key,
      rules: rules,
      absenceWindowMs: _windowMs,
      requirePersonForAbsence: _requirePerson,
    );
    Navigator.pop(context, newRules);
  }
}

class _RulesLegend extends StatelessWidget {
  const _RulesLegend();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Como cada regra funciona:',
            style: TextStyle(fontWeight: FontWeight.w600, fontSize: 12),
          ),
          SizedBox(height: 6),
          _LegendRow(
            icon: Icons.block,
            color: Colors.grey,
            title: 'Ignorar',
            subtitle: 'A classe nao influencia o alerta.',
          ),
          SizedBox(height: 4),
          _LegendRow(
            icon: Icons.add_alert,
            color: Color(0xFFE53935),
            title: 'Se detectar  ->  infracao',
            subtitle: 'Disparar quando a classe aparecer no frame.',
          ),
          SizedBox(height: 4),
          _LegendRow(
            icon: Icons.event_busy,
            color: Color(0xFFEF6C00),
            title: 'Se faltar  ->  infracao',
            subtitle: 'Disparar quando a classe sumir alem da janela.',
          ),
        ],
      ),
    );
  }
}

class _LegendRow extends StatelessWidget {
  const _LegendRow({
    required this.icon,
    required this.color,
    required this.title,
    required this.subtitle,
  });

  final IconData icon;
  final Color color;
  final String title;
  final String subtitle;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 18, color: color),
        const SizedBox(width: 8),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title,
                  style: const TextStyle(
                      fontSize: 12, fontWeight: FontWeight.w600)),
              Text(subtitle,
                  style: const TextStyle(fontSize: 11, color: Colors.black54)),
            ],
          ),
        ),
      ],
    );
  }
}

class _ClassRuleTile extends StatelessWidget {
  const _ClassRuleTile({
    required this.classLabel,
    required this.current,
    required this.onChanged,
  });

  final String classLabel;
  final RuleKind? current;
  final ValueChanged<RuleKind?> onChanged;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            classLabel,
            style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14),
          ),
          const SizedBox(height: 4),
          SizedBox(
            width: double.infinity,
            child: SegmentedButton<RuleKind?>(
              segments: const [
                ButtonSegment<RuleKind?>(
                  value: null,
                  icon: Icon(Icons.block, size: 16),
                  label: Text('Ignorar', style: TextStyle(fontSize: 12)),
                ),
                ButtonSegment<RuleKind?>(
                  value: RuleKind.presence,
                  icon: Icon(Icons.add_alert, size: 16),
                  label: Text('Se detectar', style: TextStyle(fontSize: 12)),
                ),
                ButtonSegment<RuleKind?>(
                  value: RuleKind.absence,
                  icon: Icon(Icons.event_busy, size: 16),
                  label: Text('Se faltar', style: TextStyle(fontSize: 12)),
                ),
              ],
              selected: {current},
              showSelectedIcon: false,
              style: const ButtonStyle(
                visualDensity: VisualDensity.compact,
                tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              ),
              onSelectionChanged: (s) => onChanged(s.first),
            ),
          ),
        ],
      ),
    );
  }
}
