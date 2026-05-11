import 'package:flutter/material.dart';

import '../../models/infraction_rule.dart';
import '../../models/model_descriptor.dart';
import '../../services/infraction_rules_storage.dart';
import '../../services/model_registry.dart';
import '../widgets/infraction_config_sheet.dart';
import 'camera_inference_screen.dart';
import 'storage_inference_screen.dart';

/// Menu principal da POC. Tres botoes:
/// - Camera em tempo real (deteccao + alerta de infracao)
/// - Inferencia em armazenamento (batch + JSON com metricas)
/// - Configurar regras EPI (atalho do modelo padrao)
class MenuScreen extends StatelessWidget {
  const MenuScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('POC EPI - Monitoramento')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 120,
                height: 120,
                decoration: BoxDecoration(
                  color: const Color(0xFF1976D2),
                  borderRadius: BorderRadius.circular(24),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.15),
                      blurRadius: 12,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: const Icon(
                  Icons.health_and_safety,
                  size: 72,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                'POC EPI Monitor',
                style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 32),
              _navButton(
                context,
                icon: Icons.videocam,
                label: 'Camera em tempo real',
                screen: const CameraInferenceScreen(),
              ),
              const SizedBox(height: 16),
              _navButton(
                context,
                icon: Icons.photo_library,
                label: 'Inferencia em armazenamento',
                screen: const StorageInferenceScreen(),
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                height: 56,
                child: OutlinedButton.icon(
                  onPressed: () => _openDefaultRulesSheet(context),
                  icon: const Icon(Icons.rule),
                  label: const Text('Configurar regras EPI',
                      style: TextStyle(fontSize: 18)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _openDefaultRulesSheet(BuildContext context) async {
    final descriptor =
        ModelDescriptor.fromBundled(ModelRegistry.instance.defaultModel);
    final current = InfractionRulesStorage.instance
        .ruleSetFor(descriptor.key, descriptor.defaultRules);

    final result = await showModalBottomSheet<InfractionRuleSet>(
      context: context,
      isScrollControlled: true,
      builder: (_) => InfractionConfigSheet(
        model: descriptor,
        current: current,
      ),
    );
    if (result == null) return;

    if (result == descriptor.defaultRules) {
      await InfractionRulesStorage.instance.reset(descriptor.key);
    } else {
      await InfractionRulesStorage.instance.override(result);
    }
  }

  Widget _navButton(
    BuildContext context, {
    required IconData icon,
    required String label,
    required Widget screen,
  }) {
    return SizedBox(
      width: double.infinity,
      height: 56,
      child: ElevatedButton.icon(
        onPressed: () => Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => screen),
        ),
        icon: Icon(icon),
        label: Text(label, style: const TextStyle(fontSize: 18)),
      ),
    );
  }
}
