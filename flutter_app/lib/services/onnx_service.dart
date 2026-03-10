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
/// Optimized for Latency via Parallel Pipeline and CPU Thread Tuning.
/// Developed by Shah Nawaz.
class OnnxService {
  static const int inputSize = 224;
  static const int numViews = 6;
  static const int numChannels = 3;
  static const String modelAssetPath = 'assets/models/jaaltaka_attention.onnx';

  static const List<double> mean = [0.485, 0.456, 0.406];
  static const List<double> std = [0.229, 0.224, 0.225];

  static const List<String> viewNames = [
    'View 1 - Front', 'View 2 - Back', 'View 3 - Watermark',
    'View 4 - Security Thread', 'View 5 - Serial Number', 'View 6 - Hologram / UV',
  ];

  static const List<double> viewImportance = [
    0.2990, 0.1211, 0.2655, 0.2461, 0.2104, 0.2996,
  ];

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
      
      // Standard CPU optimizations that work on all versions
      sessionOptions.setIntraOpNumThreads(2); 
      sessionOptions.setInterOpNumThreads(2);

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
    if (await modelFile.exists()) return modelFile.path;

    onProgress?.call(0.1);
    final ByteData data = await rootBundle.load(modelAssetPath);
    final List<int> bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await modelFile.writeAsBytes(bytes);
    onProgress?.call(1.0);
    return modelFile.path;
  }

  Future<AuthenticationResult> runInference(List<String> viewPaths) async {
    if (!_isLoaded || _session == null) throw Exception('Model not loaded.');

    final totalStopwatch = Stopwatch()..start();

    // 1. Decode images on Main Thread
    final List<Uint8List> rgbaList = [];
    for (var path in viewPaths) {
      rgbaList.add(await _decodeNative(path));
    }

    // 2. Heavy Normalization in Isolate
    final inputTensor = await compute(_normalizeInIsolate, rgbaList);

    // 3. Parallel Execution (Model + OCR)
    final mlFuture = _runModelInInference(inputTensor);
    final ocrFuture = _extractSerialNumber(viewPaths[4]);

    final results = await Future.wait([mlFuture, ocrFuture]);
    
    final List<double> probs = results[0] as List<double>;
    final String serialNumber = results[1] as String;

    totalStopwatch.stop();

    return AuthenticationResult(
      isAuthentic: probs[1] > probs[0],
      confidence: probs[1] > probs[0] ? probs[1] : probs[0],
      classProbabilities: {'Fake': probs[0], 'Real': probs[1]},
      inferenceTimeMs: totalStopwatch.elapsedMilliseconds.toDouble(),
      viewResults: List.generate(numViews, (i) => ViewResult(
        name: viewNames[i], importance: viewImportance[i], imagePath: viewPaths[i],
      )),
      serialNumber: serialNumber,
    );
  }

  Future<List<double>> _runModelInInference(Float32List inputTensor) async {
    final shape = [1, numViews, numChannels, inputSize, inputSize];
    final inputOrt = OrtValueTensor.createTensorWithDataList(inputTensor, shape);
    final outputs = _session!.run(OrtRunOptions(), {'views': inputOrt});
    final logits = _parseLogits(outputs[0]?.value);
    inputOrt.release();
    for (var o in outputs) o?.release();
    return _softmax(logits);
  }

  Future<Uint8List> _decodeNative(String path) async {
    final bytes = await File(path).readAsBytes();
    final buffer = await ui.ImmutableBuffer.fromUint8List(bytes);
    final descriptor = await ui.ImageDescriptor.encoded(buffer);
    final codec = await descriptor.instantiateCodec(targetWidth: inputSize, targetHeight: inputSize);
    final frame = await codec.getNextFrame();
    final image = frame.image;
    final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    final result = byteData!.buffer.asUint8List();
    image.dispose();
    descriptor.dispose();
    buffer.dispose();
    return result;
  }

  static Float32List _normalizeInIsolate(List<Uint8List> rgbaList) {
    final tensor = Float32List(numViews * numChannels * inputSize * inputSize);
    for (int v = 0; v < rgbaList.length; v++) {
      final rgba = rgbaList[v];
      final base = v * numChannels * inputSize * inputSize;
      for (int i = 0; i < inputSize * inputSize; i++) {
        final r = rgba[i * 4] / 255.0;
        final g = rgba[i * 4 + 1] / 255.0;
        final b = rgba[i * 4 + 2] / 255.0;
        tensor[base + 0 * inputSize * inputSize + i] = (r - mean[0]) / std[0];
        tensor[base + 1 * inputSize * inputSize + i] = (g - mean[1]) / std[1];
        tensor[base + 2 * inputSize * inputSize + i] = (b - mean[2]) / std[2];
      }
    }
    return tensor;
  }

  List<double> _parseLogits(dynamic outputTensor) {
    if (outputTensor is List<List<double>>) return outputTensor[0];
    if (outputTensor is List<List<num>>) return outputTensor[0].map((e) => e.toDouble()).toList();
    if (outputTensor is List && outputTensor.isNotEmpty && outputTensor[0] is List) {
      return (outputTensor[0] as List).map((e) => (e as num).toDouble()).toList();
    }
    return (outputTensor as List).map((e) => (e as num).toDouble()).toList();
  }

  List<double> _softmax(List<double> logits) {
    if (logits.isEmpty) return [0.0, 0.0];
    final maxLogit = logits.reduce(max);
    final expValues = logits.map((l) => exp(l - maxLogit)).toList();
    final expSum = expValues.reduce((a, b) => a + b);
    return expValues.map((e) => e / expSum).toList();
  }

  Future<OcclusionResult> runOcclusionSensitivity(
    List<String> viewPaths, {
    void Function(int current, int total)? onProgress,
  }) async {
    if (!_isLoaded || _session == null) throw Exception('Model not loaded.');
    final List<Uint8List> rgbaList = [];
    for (var path in viewPaths) rgbaList.add(await _decodeNative(path));
    final Float32List baselineTensor = await compute(_normalizeInIsolate, rgbaList);
    final baselineProbs = await _runModelInInference(baselineTensor);
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
          final occProbs = await _runModelInInference(workingTensor);
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
          if (completedOps % 20 == 0) {
            onProgress?.call(completedOps, totalOps);
            await Future.delayed(const Duration(milliseconds: 1));
          }
        }
      }
      heatmaps.add(grid);
    }
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
    return OcclusionResult(heatmaps: heatmaps, predictionIndex: predIdx, baselineConfidence: baselineConf, timeMs: 0.0);
  }

  Future<String> _extractSerialNumber(String imagePath) async {
    try {
      final inputImage = InputImage.fromFilePath(imagePath);
      final textRecognizer = TextRecognizer(script: TextRecognitionScript.latin);
      final RecognizedText recognizedText = await textRecognizer.processImage(inputImage);
      await textRecognizer.close();
      String bestMatch = '';
      final serialRegex = RegExp(r'[0-9০-৯]');
      for (var block in recognizedText.blocks) {
        for (var line in block.lines) {
          final text = line.text.trim();
          if (text.contains(serialRegex) && text.length >= 4 && text.length > bestMatch.length) {
            bestMatch = text;
          }
        }
      }
      return bestMatch.isNotEmpty ? bestMatch : 'Unknown';
    } catch (e) { return 'Unknown'; }
  }

  void dispose() {
    _session?.release();
    OrtEnv.instance.release();
  }
}

class OcclusionResult {
  final List<List<List<double>>> heatmaps;
  final int predictionIndex;
  final double baselineConfidence;
  final double timeMs;
  OcclusionResult({required this.heatmaps, required this.predictionIndex, required this.baselineConfidence, required this.timeMs});
}
