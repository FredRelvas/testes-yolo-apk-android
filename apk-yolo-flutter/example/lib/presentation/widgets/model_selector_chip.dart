import 'package:flutter/material.dart';

import '../../models/model_descriptor.dart';
import '../../services/model_registry.dart';
import '../../services/model_picker.dart';

/// Chip que mostra o modelo atualmente carregado e abre um bottom
/// sheet com a lista de modelos bundled + a opcao "Carregar do
/// dispositivo" (file_picker).
class ModelSelectorChip extends StatelessWidget {
  const ModelSelectorChip({
    super.key,
    required this.current,
    required this.isLoading,
    required this.onSelected,
  });

  final ModelDescriptor current;
  final bool isLoading;
  final ValueChanged<ModelDescriptor> onSelected;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.black.withValues(alpha: 0.55),
      borderRadius: BorderRadius.circular(20),
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: isLoading ? null : () => _openPicker(context),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                current.source == ModelSource.custom
                    ? Icons.folder_open
                    : Icons.memory,
                color: Colors.white,
                size: 16,
              ),
              const SizedBox(width: 6),
              Text(
                current.label,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(width: 4),
              Icon(
                isLoading ? Icons.hourglass_empty : Icons.expand_more,
                color: Colors.white,
                size: 16,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _openPicker(BuildContext context) async {
    final selected = await showModalBottomSheet<ModelDescriptor>(
      context: context,
      backgroundColor: Theme.of(context).colorScheme.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (ctx) => _ModelPickerSheet(current: current),
    );
    if (selected != null) onSelected(selected);
  }
}

class _ModelPickerSheet extends StatefulWidget {
  const _ModelPickerSheet({required this.current});

  final ModelDescriptor current;

  @override
  State<_ModelPickerSheet> createState() => _ModelPickerSheetState();
}

class _ModelPickerSheetState extends State<_ModelPickerSheet> {
  bool _picking = false;

  @override
  Widget build(BuildContext context) {
    final bundled = ModelRegistry.instance.all;
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 40,
              height: 4,
              margin: const EdgeInsets.only(bottom: 12),
              decoration: BoxDecoration(
                color: Colors.grey,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  'Selecionar modelo',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
                ),
              ),
            ),
            const SizedBox(height: 8),
            Flexible(
              child: ListView(
                shrinkWrap: true,
                children: [
                  for (final m in bundled)
                    ListTile(
                      leading: const Icon(Icons.memory),
                      title: Text(m.label),
                      subtitle: Text(m.file,
                          style: const TextStyle(fontSize: 11)),
                      trailing: widget.current.key == 'bundled:${m.file}'
                          ? const Icon(Icons.check, color: Colors.green)
                          : null,
                      onTap: () => Navigator.pop(
                        context,
                        ModelDescriptor.fromBundled(m),
                      ),
                    ),
                  const Divider(height: 1),
                  ListTile(
                    leading: const Icon(Icons.folder_open),
                    title: const Text('Carregar do dispositivo (.tflite)...'),
                    enabled: !_picking,
                    onTap: _picking ? null : _pickCustom,
                    trailing: _picking
                        ? const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : null,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _pickCustom() async {
    setState(() => _picking = true);
    try {
      final descriptor = await ModelPicker.pickFromDevice();
      if (!mounted) return;
      if (descriptor != null) {
        Navigator.pop(context, descriptor);
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Erro ao carregar modelo: $e')),
      );
    } finally {
      if (mounted) setState(() => _picking = false);
    }
  }
}
