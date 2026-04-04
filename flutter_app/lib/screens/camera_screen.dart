import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'package:vibration/vibration.dart';
import '../providers/auth_provider.dart';
import '../services/onnx_service.dart';
import 'preview_screen.dart';
import 'history_screen.dart';
import 'batch_screen.dart';

/// Screen 1: Multi-view camera capture with flash, gallery, haptic feedback.
/// Developed by Shah Nawaz.
class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isInitialized = false;
  bool _isCapturing = false;
  FlashMode _flashMode = FlashMode.off;
  final ImagePicker _picker = ImagePicker();

  static const List<IconData> _viewIcons = [
    Icons.credit_card,
    Icons.flip,
    Icons.water_drop,
    Icons.security,
    Icons.numbers,
    Icons.auto_awesome,
  ];

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras != null && _cameras!.isNotEmpty) {
        _controller = CameraController(
          _cameras!.first,
          ResolutionPreset.high,
          enableAudio: false,
          imageFormatGroup: ImageFormatGroup.jpeg,
        );
        await _controller!.initialize();
        if (mounted) {
          setState(() => _isInitialized = true);
        }
      }
    } catch (e) {
      debugPrint('Camera init error: $e');
    }
  }

  Future<void> _toggleFlash() async {
    if (_controller == null) return;
    FlashMode next;
    switch (_flashMode) {
      case FlashMode.off:
        next = FlashMode.auto;
        break;
      case FlashMode.auto:
        next = FlashMode.always;
        break;
      case FlashMode.always:
        next = FlashMode.torch;
        break;
      default:
        next = FlashMode.off;
    }
    await _controller!.setFlashMode(next);
    setState(() => _flashMode = next);
  }

  IconData get _flashIcon {
    switch (_flashMode) {
      case FlashMode.off:
        return Icons.flash_off;
      case FlashMode.auto:
        return Icons.flash_auto;
      case FlashMode.always:
        return Icons.flash_on;
      case FlashMode.torch:
        return Icons.highlight;
    }
  }

  String get _flashLabel {
    switch (_flashMode) {
      case FlashMode.off:
        return 'OFF';
      case FlashMode.auto:
        return 'AUTO';
      case FlashMode.always:
        return 'ON';
      case FlashMode.torch:
        return 'TORCH';
    }
  }

  Future<void> _hapticFeedback() async {
    try {
      final hasVibrator = await Vibration.hasVibrator();
      if (hasVibrator) {
        Vibration.vibrate(duration: 50);
      }
    } catch (_) {
      HapticFeedback.mediumImpact();
    }
  }

  Future<void> _captureView() async {
    if (_controller == null || _isCapturing) return;
    final provider = context.read<AuthenticationProvider>();
    if (provider.allViewsCaptured) return;

    setState(() => _isCapturing = true);

    try {
      final XFile photo = await _controller!.takePicture();
      provider.addView(photo.path);
      await _hapticFeedback();

      if (provider.allViewsCaptured && mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => const PreviewScreen()),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Capture failed: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isCapturing = false);
    }
  }

  Future<void> _pickFromGallery() async {
    final provider = context.read<AuthenticationProvider>();
    if (provider.allViewsCaptured) return;

    final remaining = OnnxService.numViews - provider.capturedViews.length;
    final images = await _picker.pickMultiImage(limit: remaining);

    if (images.isNotEmpty) {
      for (final img in images) {
        if (provider.capturedViews.length < OnnxService.numViews) {
          provider.addView(img.path);
        }
      }
      await _hapticFeedback();
      if (provider.allViewsCaptured && mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => const PreviewScreen()),
        );
      }
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('JaalTaka'),
        actions: [
          // Flash toggle
          if (_isInitialized)
            IconButton(
              icon: Icon(_flashIcon),
              onPressed: _toggleFlash,
              tooltip: 'Flash: $_flashLabel',
            ),
          // History
          IconButton(
            icon: const Icon(Icons.history),
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const HistoryScreen()),
            ),
            tooltip: 'Scan history',
          ),
          // Batch Mode
          IconButton(
            icon: const Icon(Icons.library_books),
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const BatchScreen()),
            ),
            tooltip: 'Batch Analysis',
          ),
          // More Menu
          PopupMenuButton<String>(
            onSelected: (value) {
              final provider = context.read<AuthenticationProvider>();
              if (value == 'theme') {
                provider.toggleTheme();
              } else if (value == 'lang') {
                provider.toggleLanguage();
              } else if (value == 'voice') {
                provider.toggleVoice();
              } else if (value == 'model') {
                provider.toggleModelMode();
              }
            },
            itemBuilder: (BuildContext context) {
              final provider = context.read<AuthenticationProvider>();
              final themeIcon = provider.themeMode == ThemeMode.dark
                  ? Icons.light_mode
                  : provider.themeMode == ThemeMode.light
                      ? Icons.dark_mode
                      : Icons.brightness_auto;

              return [
                PopupMenuItem(
                  value: 'model',
                  child: Row(
                    children: [
                      Icon(
                          provider.useFastModel
                              ? Icons.bolt
                              : Icons.precision_manufacturing,
                          size: 20,
                          color: Colors.amber),
                      const SizedBox(width: 8),
                      Text(provider.useFastModel
                          ? 'Mode: Fast (INT8)'
                          : 'Mode: Accurate (FP32)'),
                    ],
                  ),
                ),
                PopupMenuItem(
                  value: 'theme',
                  child: Row(
                    children: [
                      Icon(themeIcon, size: 20),
                      const SizedBox(width: 8),
                      const Text('Toggle Theme'),
                    ],
                  ),
                ),
                PopupMenuItem(
                  value: 'lang',
                  child: Row(
                    children: [
                      const Icon(Icons.language, size: 20),
                      const SizedBox(width: 8),
                      Text(provider.isBangla
                          ? 'Switch to English'
                          : 'বাংলায় পরিবর্তন করুন'),
                    ],
                  ),
                ),
                PopupMenuItem(
                  value: 'voice',
                  child: Row(
                    children: [
                      Icon(
                          provider.isVoiceEnabled
                              ? Icons.volume_up
                              : Icons.volume_off,
                          size: 20),
                      const SizedBox(width: 8),
                      Text(provider.isVoiceEnabled
                          ? 'Mute Voice'
                          : 'Enable Voice'),
                    ],
                  ),
                ),
              ];
            },
          ),
        ],
      ),
      body: Consumer<AuthenticationProvider>(
        builder: (_, provider, __) {
          final viewIdx = provider.currentViewIndex;
          final captured = provider.capturedViews.length;
          final total = OnnxService.numViews;

          // Model downloading overlay
          if (!provider.isModelLoaded && provider.modelLoadProgress > 0) {
            return Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.cloud_download,
                      size: 60, color: Colors.blue),
                  const SizedBox(height: 16),
                  Text(
                    provider.isBangla
                        ? 'এআই মডেল ডাউনলোড হচ্ছে...'
                        : 'Downloading AI Model...',
                    style: const TextStyle(
                        fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 16),
                  SizedBox(
                    width: 200,
                    child: LinearProgressIndicator(
                        value: provider.modelLoadProgress),
                  ),
                  const SizedBox(height: 8),
                  Text(
                      '${(provider.modelLoadProgress * 100).toStringAsFixed(0)}%'),
                ],
              ),
            );
          }

          return Stack(
            children: [
              // Camera preview
              if (_isInitialized && _controller != null)
                Positioned.fill(child: CameraPreview(_controller!))
              else
                const Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      CircularProgressIndicator(),
                      SizedBox(height: 16),
                      Text('Initializing camera...'),
                    ],
                  ),
                ),

              // Top: Progress indicator
              Positioned(
                top: 16,
                left: 16,
                right: 16,
                child: Card(
                  color: Colors.black.withValues(alpha: 0.7),
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      children: [
                        Text(
                          provider.isBangla
                              ? 'ছবি $captured / $total'
                              : 'View $captured / $total',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 8),
                        // Progress dots
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: List.generate(total, (i) {
                            final completed = i < captured;
                            final current = i == captured;
                            return Container(
                              margin: const EdgeInsets.symmetric(horizontal: 4),
                              width: current ? 14 : 10,
                              height: current ? 14 : 10,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: completed
                                    ? Colors.green
                                    : current
                                        ? Colors.amber
                                        : Colors.grey,
                                border: current
                                    ? Border.all(color: Colors.white, width: 2)
                                    : null,
                              ),
                            );
                          }),
                        ),
                      ],
                    ),
                  ),
                ),
              ),

              // Bottom: View instruction with icon
              Positioned(
                bottom: 130,
                left: 16,
                right: 16,
                child: Card(
                  color: Colors.black.withValues(alpha: 0.75),
                  child: Padding(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 14,
                    ),
                    child: Column(
                      children: [
                        if (viewIdx < total) ...[
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                _viewIcons[viewIdx],
                                color: Colors.amber,
                                size: 20,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                OnnxService.viewNames[viewIdx],
                                style: const TextStyle(
                                  color: Colors.amber,
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 4),
                          Text(
                            provider.currentInstruction,
                            textAlign: TextAlign.center,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 14,
                            ),
                          ),
                        ] else
                          Text(
                            provider.currentInstruction,
                            style: const TextStyle(
                              color: Colors.green,
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
              ),

              // Error display
              if (provider.error != null)
                Positioned(
                  top: 100,
                  left: 16,
                  right: 16,
                  child: Card(
                    color: Colors.red.shade100,
                    child: Padding(
                      padding: const EdgeInsets.all(8),
                      child: Text(
                        provider.error!,
                        style: const TextStyle(color: Colors.red),
                      ),
                    ),
                  ),
                ),
            ],
          );
        },
      ),

      // Capture, gallery, undo buttons
      floatingActionButton: _isInitialized
          ? Consumer<AuthenticationProvider>(
              builder: (_, provider, __) {
                return Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    // Undo last capture
                    if (provider.capturedViews.isNotEmpty &&
                        !provider.allViewsCaptured)
                      Padding(
                        padding: const EdgeInsets.only(right: 16),
                        child: FloatingActionButton(
                          heroTag: 'undo',
                          onPressed: provider.removeLastView,
                          backgroundColor: Colors.orange,
                          child: const Icon(Icons.undo, color: Colors.white),
                        ),
                      ),

                    // Gallery pick button
                    if (!provider.allViewsCaptured)
                      Padding(
                        padding: const EdgeInsets.only(right: 16),
                        child: FloatingActionButton(
                          heroTag: 'gallery',
                          onPressed: _pickFromGallery,
                          backgroundColor: Colors.blue,
                          child: const Icon(Icons.photo_library,
                              color: Colors.white),
                        ),
                      ),

                    // Capture / Continue button
                    FloatingActionButton.large(
                      heroTag: 'capture',
                      onPressed: _isCapturing
                          ? null
                          : provider.allViewsCaptured
                              ? () {
                                  Navigator.push(
                                    context,
                                    MaterialPageRoute(
                                      builder: (_) => const PreviewScreen(),
                                    ),
                                  );
                                }
                              : _captureView,
                      child: _isCapturing
                          ? const CircularProgressIndicator(color: Colors.white)
                          : provider.allViewsCaptured
                              ? const Icon(Icons.arrow_forward, size: 36)
                              : const Icon(Icons.camera_alt, size: 36),
                    ),
                  ],
                );
              },
            )
          : null,
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}
