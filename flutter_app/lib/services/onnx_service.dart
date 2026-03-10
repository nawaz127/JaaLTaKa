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
/// Enhanced with Hybrid Preprocessing: Native Decoding (Main) + Normalization (Isolate).
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

  /// Run inference on 6 view images.
  Future<AuthenticationResult> runInference(List<String> viewPaths) async {
    if (!_isLoaded || _session == null) throw Exception('Model not loaded.');

    final stopwatch = Stopwatch()..start();

    // 1. Decode images on Main Thread (Safe for native decoders)
    final List<Uint8List> rgbaList = [];
    for (var path in viewPaths) {
      rgbaList.add(await _decodeNative(path));
    }

    // 2. Heavy Normalization in Isolate (Safe for computation)
    final inputTensor = await compute(_normalizeInIsolate, rgbaList);

    final shape = [1, numViews, numChannels, inputSize, inputSize];
    final inputOrt = OrtValueTensor.createTensorWithDataList(inputTensor, shape);
    final outputs = _session!.run(OrtRunOptions(), {'views': inputOrt});
    
    final logits = _parseLogits(outputs[0]?.value);
    final probs = _softmax(logits);
    
    inputOrt.release();
    for (var o in outputs) o?.release();

    final serialNumber = await _extractSerialNumber(viewPaths[4]);
    stopwatch.stop();

    return AuthenticationResult(
      isAuthentic: probs[1] > probs[0],
      confidence: probs[1] > probs[0] ? probs[1] : probs[0],
      classProbabilities: {'Fake': probs[0], 'Real': probs[1]},
      inferenceTimeMs: stopwatch.elapsedMilliseconds.toDouble(),
      viewResults: List.generate(numViews, (i) => ViewResult(
        name: viewNames[i], importance: viewImportance[i], imagePath: viewPaths[i],
      )),
      serialNumber: serialNumber,
    );
  }

  /// Native efficient decoding on Main Thread.
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

  /// Normalization logic moved to background to keep UI smooth.
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
    return (outputTensor as List).map((e) => (e as num).toDouble()).toList();
  }

  List<double> _softmax(List<double> logits) {
    final maxLogit = logits.reduce(max);
    final expValues = logits.map((l) => exp(l - maxLogit)).toList();
    final expSum = expValues.reduce((a, b) => a + b);
    return expValues.map((e) => e / expSum).toList();
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
