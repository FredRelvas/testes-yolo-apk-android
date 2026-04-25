// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo_example/presentation/screens/menu_screen.dart';
import 'package:ultralytics_yolo_example/services/infraction_rules_storage.dart';
import 'package:ultralytics_yolo_example/services/model_registry.dart';
import 'package:ultralytics_yolo_example/services/system_metrics_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ModelRegistry.load();
  await InfractionRulesStorage.load();
  await SystemMetricsService.instance.start();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'POC EPI - Monitoramento',
      home: MenuScreen(),
    );
  }
}
