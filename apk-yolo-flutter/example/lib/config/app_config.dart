// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

/// Runtime configuration injected at build time via `--dart-define`.
///
/// Usage:
///   flutter run --dart-define=USE_GPU=true     # enable GPU delegate
///   flutter run                                # GPU disabled (default)
///
/// The default is `false` to avoid GPU OOM errors on devices where the full
/// model cannot fit in GPU memory (e.g. COCO fp32 with 690 delegated nodes).
const bool kUseGpu = bool.fromEnvironment('USE_GPU', defaultValue: false);
