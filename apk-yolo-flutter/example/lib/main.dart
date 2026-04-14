// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo_example/presentation/screens/home_screen.dart';
import 'package:ultralytics_yolo_example/services/model_registry.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ModelRegistry.load();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'YOLO useGpu Example',
      home: HomeScreen(),
    );
  }
}
