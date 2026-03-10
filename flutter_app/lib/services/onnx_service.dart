import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path_provider/path_provider.dart';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';

import '../models/auth_result.dart';

/// ONNX Runtime inference service for banknote authentication.
/// Enhanced with memory-efficient preprocessing and Isolate support.
/// Developed by Shah Nawaz.
class OnnxService {
  static const int inputSize = 224;
  static const int numViews = 6;
  static const int numChannels = 3;
  static const String modelAssetPath = 'assets/models/jaaltaka_attention.onnx';

  // ImageNet normalization constants
  static const List<double> mean = [0.485, 0.456, 0.406];
  static const List<double> std = [0.229, 0.224, 0.225];

  // View labels for the guided capture
  static const List<String> viewNames = [
    'View 1 - Front',
    'View 2 - Back',
    'View 3 - Watermark',
    'View 4 - Security Thread',
    'View 5 - Serial Number',
    'View 6 - Hologram / UV',
  ];

  // SHAP-based importance for each view (from training analysis)
  static const List<double> viewImportance = [
    0.2990, 0.1211, 0.2655, 0.2461, 0.2104, 0.2996,
  ];

  // Occlusion heatmap grid size (7×7 = 49 cells per view)
  static const int occlusionGridSize = 7;

  OrtSession? _session;
  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;

  void init() {
    OrtEnv.instance.init();
  }

  Future<void> loadModel({void Function(double)? onProgress}) async {
    if (_isLoaded) return;
    try {
      final modelPath = await _copyAssetToLocal(onProgress);
      final sessionOptions = OrtSessionOptions();
      _session = OrtSession.fromFile(File(modelPath), sessionOptions);
      _isLoaded = true;
    } catch (e) {
      _isLoaded = false;
      debugPrint('Error loading model: $e');
      rethrow;
    }
  }

  Future<String> _copyAssetToLocal(void Function(double)? onProgress) async {
    final appDir = await getApplicationDocumentsDirectory();
    final modelFile = File('${appDir.path}/jaaltaka_attention.onnx');

    if (await modelFile.exists()) {
      return modelFile.path;
    }

    onProgress?.call(0.1);
    final ByteData data = await rootBundle.load(modelAssetPath);
    final List<int> bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await modelFile.writeAsBytes(bytes);
    onProgress?.call(1.0);
    
    return modelFile.path;
  }

  /// Parse ONNX output safely handling List<List<double>>, List<List<num>>, flat list.
  List<double> _parseLogits(dynamic outputTensor) {
    if (outputTensor is List<List<double>>) {
      return outputTensor[0];
    } else if (outputTensor is List<List<num>>) {
      return outputTensor[0].map((e) => e.toDouble()).toList();
    } else if (outputTensor is List) {
      if (outputTensor.isNotEmpty && outputTensor[0] is List) {
        return (outputTensor[0] as List).map((e) => (e as num).toDouble()).toList();
      }
      return outputTensor.map((e) => (e as num).toDouble()).toList();
    }
    throw Exception('Unexpected output format: ${outputTensor.runtimeType}');
  }

  /// Apply softmax to logits.
  List<double> _softmax(List<double> logits) {
    final maxLogit = logits.reduce(max);
    final expValues = logits.map((l) => exp(l - maxLogit)).toList();
    final expSum = expValues.reduce((a, b) => a + b);
    return expValues.map((e) => e / expSum).toList();
  }

  /// Run single inference and return raw probabilities.
  List<double> _runRawInference(Float32List inputTensor) {
    final shape = [1, numViews, numChannels, inputSize, inputSize];
    final inputOrt = OrtValueTensor.createTensorWithDataList(inputTensor, shape);
    final runOptions = OrtRunOptions();
    final outputs = _session!.run(runOptions, {'views': inputOrt});
    final logits = _parseLogits(outputs[0]?.value);
    final probs = _softmax(logits);
    inputOrt.release();
    runOptions.release();
    for (final o in outputs) {
      o?.release();
    }
    return probs;
  }

  /// Run inference on 6 view images. Returns [AuthenticationResult].
  Future<AuthenticationResult> runInference(List<String> viewPaths) async {
    if (!_isLoaded || _session == null) {
      throw Exception('Model not loaded. Call loadModel() first.');
    }

    final stopwatch = Stopwatch()..start();

    // PERFORMANCE: Preprocess images in a separate isolate
    final inputTensor = await compute(_preprocessViewsStatic, viewPaths);

    final probs = _runRawInference(inputTensor);
    final serialNumber = await _extractSerialNumber(viewPaths[4]);

    final viewResults = List.generate(numViews, (i) {
      return ViewResult(
        name: viewNames[i],
        importance: viewImportance[i],
        imagePath: viewPaths[i],
      );
    });

    stopwatch.stop();

    return AuthenticationResult(
      isAuthentic: probs[1] > probs[0],
      confidence: probs[1] > probs[0] ? probs[1] : probs[0],
      classProbabilities: {'Fake': probs[0], 'Real': probs[1]},
      inferenceTimeMs: stopwatch.elapsedMilliseconds.toDouble(),
      viewResults: viewResults,
      serialNumber: serialNumber,
    );
  }

  /// OCR with Bengali numeral support.
  Future<String> _extractSerialNumber(String imagePath) async {
    try {
      final inputImage = InputImage.fromFilePath(imagePath);
      final textRecognizer = TextRecognizer(script: TextRecognitionScript.latin);
      final RecognizedText recognizedText = await textRecognizer.processImage(inputImage);
      await textRecognizer.close();

      String bestMatch = '';
      final serialRegex = RegExp(r'[0-9০-৯]');
      
      for (TextBlock block in recognizedText.blocks) {
        for (TextLine line in block.lines) {
          final text = line.text.trim();
          if (text.contains(serialRegex) && text.length >= 4) {
            if (text.length > bestMatch.length) {
              bestMatch = text;
            }
          }
        }
      }
      return bestMatch.isNotEmpty ? bestMatch : 'Unknown';
    } catch (e) {
      debugPrint('OCR Error: $e');
      return 'Unknown';
    }
  }

  /// Generate occlusion sensitivity heatmaps for all 6 views.
  Future<OcclusionResult> runOcclusionSensitivity(
    List<String> viewPaths, {
    void Function(int current, int total)? onProgress,
  }) async {
    if (!_isLoaded || _session == null) throw Exception('Model not loaded.');

    final stopwatch = Stopwatch()..start();
    
    // We use compute for preprocessing but run inference on main thread for session access
    final Float32List baselineTensor = await compute(_preprocessViewsStatic, viewPaths);
    final baselineProbs = _runRawInference(baselineTensor);
    final predIdx = baselineProbs[1] > baselineProbs[0] ? 1 : 0;
    final baselineConf = baselineProbs[predIdx];

    final cellSize = inputSize ~/ occlusionGridSize;
    final totalOps = numViews * occlusionGridSize * occlusionGridSize;
    int completedOps = 0;

    final workingTensor = Float32List.fromList(baselineTensor);
    final backup = Float32List(numChannels * cellSize * cellSize);
    final heatmaps = <List<List<double>>>[];

    for (int v = 0; v < numViews; v++) {
      final grid = List.generate(occlusionGridSize, (_) => List.filled(occlusionGridSize, 0.0));
      final viewBase = v * numChannels * inputSize * inputSize;

      for (int gy = 0; gy < occlusionGridSize; gy++) {
        for (int gx = 0; gx < occlusionGridSize; gx++) {
          final yStart = gy * cellSize;
          final xStart = gx * cellSize;
          final yEnd = (gy == occlusionGridSize - 1) ? inputSize : yStart + cellSize;
          final xEnd = (gx == occlusionGridSize - 1) ? inputSize : xStart + cellSize;

          int bIdx = 0;
          for (int c = 0; c < numChannels; c++) {
            final cBase = viewBase + c * inputSize * inputSize;
            for (int y = yStart; y < yEnd; y++) {
              for (int x = xStart; x < xEnd; x++) {
                final idx = cBase + y * inputSize + x;
                backup[bIdx++] = workingTensor[idx];
                workingTensor[idx] = 0.0; 
              }
            }
          }

          final occProbs = _runRawInference(workingTensor);
          grid[gy][gx] = (baselineConf - occProbs[predIdx]).clamp(0.0, 1.0);

          bIdx = 0;
          for (int c = 0; c < numChannels; c++) {
            final cBase = viewBase + c * inputSize * inputSize;
            for (int y = yStart; y < yEnd; y++) {
              for (int x = xStart; x < xEnd; x++) {
                workingTensor[cBase + y * inputSize + x] = backup[bIdx++];
              }
            }
          }

          completedOps++;
          if (completedOps % 10 == 0) {
            onProgress?.call(completedOps, totalOps);
            await Future.delayed(const Duration(milliseconds: 1));
          }
        }
      }
      heatmaps.add(grid);
    }

    // Normalize
    for (int v = 0; v < numViews; v++) {
      double maxVal = 0.0;
      for (final row in heatmaps[v]) for (final val in row) if (val > maxVal) maxVal = val;
      if (maxVal > 0) {
        for (int r = 0; r < occlusionGridSize; r++) {
          for (int c = 0; c < occlusionGridSize; c++) {
            heatmaps[v][r][c] /= maxVal;
          }
        }
      }
    }

    stopwatch.stop();
    return OcclusionResult(
      heatmaps: heatmaps,
      predictionIndex: predIdx,
      baselineConfidence: baselineConf,
      timeMs: stopwatch.elapsedMilliseconds.toDouble(),
    );
  }

  /// Memory-efficient preprocessing in Isolate.
  static Future<Float32List> _preprocessViewsStatic(List<String> paths) async {
    final tensor = Float32List(numViews * numChannels * inputSize * inputSize);

    for (int v = 0; v < numViews; v++) {
      final bytes = await File(paths[v]).readAsBytes();
      final buffer = await ui.ImmutableBuffer.fromUint8List(bytes);
      final descriptor = await ui.ImageDescriptor.encoded(buffer);
      final codec = await descriptor.instantiateCodec(targetWidth: inputSize, targetHeight: inputSize);
      final frame = await codec.getNextFrame();
      final image = frame.image;

      final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      if (byteData == null) continue;

      final rgba = byteData.buffer.asUint8List();
      final base = v * numChannels * inputSize * inputSize;

      for (int i = 0; i < inputSize * inputSize; i++) {
        final r = rgba[i * 4] / 255.0;
        final g = rgba[i * 4 + 1] / 255.0;
        final b = rgba[i * 4 + 2] / 255.0;

        tensor[base + 0 * inputSize * inputSize + i] = (r - mean[0]) / std[0];
        tensor[base + 1 * inputSize * inputSize + i] = (g - mean[1]) / std[2];
        tensor[base + 2 * inputSize * inputSize + i] = (b - mean[2]) / std[2];
      }
      
      image.dispose();
      descriptor.dispose();
      buffer.dispose();
    }
    return tensor;
  }

  void dispose() {
    _session?.release();
    _session = null;
    _isLoaded = false;
    OrtEnv.instance.release();
  }
}

class OcclusionResult {
  final List<List<List<double>>> heatmaps;
  final int predictionIndex;
  final double baselineConfidence;
  final double timeMs;

  OcclusionResult({
    required this.heatmaps,
    required this.predictionIndex,
    required this.baselineConfidence,
    required this.timeMs,
  });
}
