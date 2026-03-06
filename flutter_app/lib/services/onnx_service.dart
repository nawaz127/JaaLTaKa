import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path_provider/path_provider.dart';

import '../models/auth_result.dart';

/// ONNX Runtime inference service for banknote authentication.
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

  Future<void> loadModel() async {
    if (_isLoaded) return;
    try {
      final modelPath = await _getModelPath();
      final sessionOptions = OrtSessionOptions();
      _session = OrtSession.fromFile(File(modelPath), sessionOptions);
      _isLoaded = true;
    } catch (e) {
      _isLoaded = false;
      rethrow;
    }
  }

  Future<String> _getModelPath() async {
    final appDir = await getApplicationDocumentsDirectory();
    final modelFile = File('${appDir.path}/jaaltaka_attention.onnx');
    if (!await modelFile.exists()) {
      final data = await rootBundle.load(modelAssetPath);
      await modelFile.writeAsBytes(data.buffer.asUint8List());
    }
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
    if (viewPaths.length != numViews) {
      throw Exception('Expected $numViews view images, got ${viewPaths.length}');
    }

    final stopwatch = Stopwatch()..start();

    // 1. Preprocess all 6 views
    final inputTensor = await _preprocessViews(viewPaths);

    // 2. Run inference
    final probs = _runRawInference(inputTensor);

    // probs[0] = Fake probability, probs[1] = Real probability
    final isAuthentic = probs[1] > probs[0];
    final confidence = isAuthentic ? probs[1] : probs[0];

    final viewResults = List.generate(numViews, (i) {
      return ViewResult(
        name: viewNames[i],
        importance: viewImportance[i],
        imagePath: viewPaths[i],
      );
    });

    stopwatch.stop();

    return AuthenticationResult(
      isAuthentic: isAuthentic,
      confidence: confidence,
      classProbabilities: {'Fake': probs[0], 'Real': probs[1]},
      inferenceTimeMs: stopwatch.elapsedMilliseconds.toDouble(),
      viewResults: viewResults,
    );
  }

  /// Generate occlusion sensitivity heatmaps for all 6 views.
  /// Divides each view into a 7×7 grid, occludes each cell, measures
  /// confidence drop → heatmap. Returns list of 6 heatmaps (7×7 grids).
  Future<OcclusionResult> runOcclusionSensitivity(
    List<String> viewPaths, {
    void Function(int current, int total)? onProgress,
  }) async {
    if (!_isLoaded || _session == null) {
      throw Exception('Model not loaded.');
    }

    final stopwatch = Stopwatch()..start();
    
    // Preprocess views once
    final Float32List baselineTensor = await _preprocessViews(viewPaths);
    final baselineProbs = _runRawInference(baselineTensor);
    final predIdx = baselineProbs[1] > baselineProbs[0] ? 1 : 0;
    final baselineConf = baselineProbs[predIdx];

    final cellSize = inputSize ~/ occlusionGridSize;
    final totalOps = numViews * occlusionGridSize * occlusionGridSize;
    int completedOps = 0;

    // We will modify the baselineTensor in-place to save memory
    final workingTensor = Float32List.fromList(baselineTensor);
    final backup = Float32List(numChannels * cellSize * cellSize);

    final heatmaps = <List<List<double>>>[];

    for (int v = 0; v < numViews; v++) {
      final grid = List.generate(
        occlusionGridSize,
        (_) => List.filled(occlusionGridSize, 0.0),
      );

      final viewBase = v * numChannels * inputSize * inputSize;

      for (int gy = 0; gy < occlusionGridSize; gy++) {
        for (int gx = 0; gx < occlusionGridSize; gx++) {
          final yStart = gy * cellSize;
          final xStart = gx * cellSize;
          final yEnd = (gy == occlusionGridSize - 1) ? inputSize : yStart + cellSize;
          final xEnd = (gx == occlusionGridSize - 1) ? inputSize : xStart + cellSize;

          // 1. Backup the original region and fill with 0 (occlusion)
          int backupIdx = 0;
          for (int c = 0; c < numChannels; c++) {
            final channelBase = viewBase + c * inputSize * inputSize;
            for (int y = yStart; y < yEnd; y++) {
              final rowBase = channelBase + y * inputSize;
              for (int x = xStart; x < xEnd; x++) {
                final idx = rowBase + x;
                backup[backupIdx++] = workingTensor[idx];
                workingTensor[idx] = 0.0; // Occlude
              }
            }
          }

          // 2. Run inference
          final occProbs = _runRawInference(workingTensor);
          final drop = baselineConf - occProbs[predIdx];
          grid[gy][gx] = drop.clamp(0.0, 1.0);

          // 3. Restore from backup
          backupIdx = 0;
          for (int c = 0; c < numChannels; c++) {
            final channelBase = viewBase + c * inputSize * inputSize;
            for (int y = yStart; y < yEnd; y++) {
              final rowBase = channelBase + y * inputSize;
              for (int x = xStart; x < xEnd; x++) {
                workingTensor[rowBase + x] = backup[backupIdx++];
              }
            }
          }

          completedOps++;
          if (completedOps % 5 == 0) {
            onProgress?.call(completedOps, totalOps);
            // Crucial: Let the UI thread breathe and Garbage Collector work
            await Future.delayed(const Duration(milliseconds: 10));
          }
        }
      }
      heatmaps.add(grid);
    }

    // Normalize each heatmap to [0,1]
    for (int v = 0; v < numViews; v++) {
      double maxVal = 0.0;
      for (final row in heatmaps[v]) {
        for (final val in row) {
          if (val > maxVal) maxVal = val;
        }
      }
      if (maxVal > 0) {
        for (int gy = 0; gy < occlusionGridSize; gy++) {
          for (int gx = 0; gx < occlusionGridSize; gx++) {
            heatmaps[v][gy][gx] /= maxVal;
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

  /// Preprocess image: decode → EXIF rotate → force uint8 → resize → normalize.
  img.Image _preprocessImage(Uint8List bytes) {
    var decoded = img.decodeImage(bytes);
    if (decoded == null) throw Exception('Failed to decode image');

    // FIX: Apply EXIF orientation (camera photos are often stored rotated)
    decoded = img.bakeOrientation(decoded);

    // FIX: Force uint8 format so pixel.r/g/b return 0-255 integers
    decoded = decoded.convert(format: img.Format.uint8, numChannels: 3);

    // Resize to 224×224
    return img.copyResize(decoded, width: inputSize, height: inputSize);
  }

  /// Preprocess 6 view images into a Float32List tensor [1, 6, 3, 224, 224].
  Future<Float32List> _preprocessViews(List<String> viewPaths) async {
    final totalSize = 1 * numViews * numChannels * inputSize * inputSize;
    final tensor = Float32List(totalSize);

    for (int v = 0; v < numViews; v++) {
      final file = File(viewPaths[v]);
      final bytes = await file.readAsBytes();
      final resized = _preprocessImage(bytes);

      // Fill tensor in CHW layout with ImageNet normalization
      final baseIdx = v * numChannels * inputSize * inputSize;
      for (int y = 0; y < inputSize; y++) {
        for (int x = 0; x < inputSize; x++) {
          final pixel = resized.getPixel(x, y);
          final r = pixel.r.toInt() / 255.0;
          final g = pixel.g.toInt() / 255.0;
          final b = pixel.b.toInt() / 255.0;

          tensor[baseIdx + 0 * inputSize * inputSize + y * inputSize + x] =
              (r - mean[0]) / std[0];
          tensor[baseIdx + 1 * inputSize * inputSize + y * inputSize + x] =
              (g - mean[1]) / std[1];
          tensor[baseIdx + 2 * inputSize * inputSize + y * inputSize + x] =
              (b - mean[2]) / std[2];
        }
      }
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

/// Result of occlusion sensitivity analysis.
class OcclusionResult {
  final List<List<List<double>>> heatmaps; // [view][row][col]
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
