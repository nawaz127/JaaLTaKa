import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_tts/flutter_tts.dart';
import '../services/onnx_service.dart';
import '../models/auth_result.dart';

/// Manages authentication state, history, theme, and model inference.
/// Developed by Shah Nawaz.
class AuthenticationProvider extends ChangeNotifier {
  final OnnxService _onnxService = OnnxService();
  final FlutterTts _tts = FlutterTts();

  bool _isModelLoaded = false;
  bool _isProcessing = false;
  bool _isGeneratingHeatmap = false;
  AuthenticationResult? _lastResult;
  OcclusionResult? _lastOcclusion;
  String? _error;
  double _modelLoadProgress = 0.0;
  int _heatmapProgress = 0;
  int _heatmapTotal = 0;

  // Preferences
  ThemeMode _themeMode = ThemeMode.system;
  bool _isBangla = false;
  bool _isVoiceEnabled = true;

  // Multi-view capture state
  final List<String> _capturedViews = [];
  int _currentViewIndex = 0;

  // History
  final List<ScanHistoryEntry> _history = [];

  // Translations
  static const Map<String, List<String>> _instructions = {
    'en': [
      'Place the FRONT of the banknote in the frame',
      'Flip over — capture the BACK of the banknote',
      'Hold up to light — capture the WATERMARK area',
      'Capture the SECURITY THREAD (metallic strip)',
      'Zoom in on the SERIAL NUMBER',
      'Capture the HOLOGRAM or special feature',
      'All views captured. Processing...'
    ],
    'bn': [
      'টাকার সামনের দিকটি ফ্রেমের মধ্যে রাখুন',
      'উল্টে দিন — পিছনের দিকটি ক্যাপচার করুন',
      'আলোর দিকে ধরুন — জলছাপ (Watermark) ক্যাপচার করুন',
      'নিরাপত্তা সুতা (Security Thread) ক্যাপচার করুন',
      'সিরিয়াল নম্বরের (Serial Number) উপর জুম করুন',
      'হলোগ্রাম (Hologram) ক্যাপচার করুন',
      'সব ছবি নেওয়া হয়েছে। প্রসেস করা হচ্ছে...'
    ]
  };

  // Getters
  bool get isModelLoaded => _isModelLoaded;
  bool get isProcessing => _isProcessing;
  bool get isGeneratingHeatmap => _isGeneratingHeatmap;
  AuthenticationResult? get lastResult => _lastResult;
  OcclusionResult? get lastOcclusion => _lastOcclusion;
  String? get error => _error;
  double get modelLoadProgress => _modelLoadProgress;
  int get heatmapProgress => _heatmapProgress;
  int get heatmapTotal => _heatmapTotal;
  ThemeMode get themeMode => _themeMode;
  bool get isBangla => _isBangla;
  bool get isVoiceEnabled => _isVoiceEnabled;
  List<String> get capturedViews => List.unmodifiable(_capturedViews);
  int get currentViewIndex => _currentViewIndex;
  bool get allViewsCaptured => _capturedViews.length >= OnnxService.numViews;
  String get currentViewName =>
      _currentViewIndex < OnnxService.viewNames.length
          ? OnnxService.viewNames[_currentViewIndex]
          : '';
  List<ScanHistoryEntry> get history => List.unmodifiable(_history);

  String get currentInstruction {
    final lang = _isBangla ? 'bn' : 'en';
    final idx = _currentViewIndex < _instructions[lang]!.length ? _currentViewIndex : _instructions[lang]!.length - 1;
    return _instructions[lang]![idx];
  }

  /// Initialize provider — load theme preference, language, and history.
  Future<void> initialize() async {
    final prefs = await SharedPreferences.getInstance();
    final themeIdx = prefs.getInt('themeMode') ?? 0;
    _themeMode = ThemeMode.values[themeIdx];
    _isBangla = prefs.getBool('isBangla') ?? false;
    _isVoiceEnabled = prefs.getBool('isVoiceEnabled') ?? true;
    
    await _initTts();
    await _loadHistory();
    notifyListeners();
  }

  Future<void> _initTts() async {
    await _tts.setLanguage(_isBangla ? 'bn-BD' : 'en-US');
    await _tts.setSpeechRate(0.5);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);
  }

  Future<void> _speak(String text) async {
    if (!_isVoiceEnabled) return;
    await _tts.stop();
    await _tts.speak(text);
  }

  void toggleLanguage() async {
    _isBangla = !_isBangla;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('isBangla', _isBangla);
    await _initTts();
    _speak(currentInstruction);
    notifyListeners();
  }

  void toggleVoice() async {
    _isVoiceEnabled = !_isVoiceEnabled;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('isVoiceEnabled', _isVoiceEnabled);
    if (!_isVoiceEnabled) {
      await _tts.stop();
    } else {
      _speak(currentInstruction);
    }
    notifyListeners();
  }

  /// Toggle theme: system → light → dark → system
  void toggleTheme() async {
    switch (_themeMode) {
      case ThemeMode.system:
        _themeMode = ThemeMode.light;
        break;
      case ThemeMode.light:
        _themeMode = ThemeMode.dark;
        break;
      case ThemeMode.dark:
        _themeMode = ThemeMode.system;
        break;
    }
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt('themeMode', _themeMode.index);
    notifyListeners();
  }

  Future<void> loadModel() async {
    try {
      _modelLoadProgress = 0.0;
      notifyListeners();
      _onnxService.init();
      
      await _onnxService.loadModel(onProgress: (progress) {
        _modelLoadProgress = progress;
        notifyListeners();
      });
      
      _modelLoadProgress = 1.0;
      _isModelLoaded = true;
      _error = null;
    } catch (e) {
      _error = 'Failed to load model: $e';
      _isModelLoaded = false;
    }
    notifyListeners();
  }

  void addView(String imagePath) {
    if (_capturedViews.length < OnnxService.numViews) {
      _capturedViews.add(imagePath);
      _currentViewIndex = _capturedViews.length;
      _error = null;
      _speak(currentInstruction);
      notifyListeners();
    }
  }

  void removeLastView() {
    if (_capturedViews.isNotEmpty) {
      _capturedViews.removeLast();
      _currentViewIndex = _capturedViews.length;
      _speak(currentInstruction);
      notifyListeners();
    }
  }

  /// Authenticate using all 6 captured views.
  Future<AuthenticationResult?> authenticateViews() async {
    if (!allViewsCaptured) {
      _error = _isBangla ? 'অনুগ্রহ করে ৬টি ছবি তুলুন।' : 'Please capture all ${OnnxService.numViews} views first.';
      notifyListeners();
      return null;
    }

    if (!_isModelLoaded) {
      await loadModel();
      if (!_isModelLoaded) return null;
    }

    _isProcessing = true;
    _error = null;
    _lastOcclusion = null;
    notifyListeners();

    try {
      final result = await _onnxService.runInference(_capturedViews);
      _lastResult = result;
      _isProcessing = false;

      // Save to history
      await _addToHistory(result);

      // Voice result
      final verdict = result.isAuthentic 
        ? (_isBangla ? 'নোটটি আসল' : 'Note is Authentic') 
        : (_isBangla ? 'নোটটি জাল' : 'Note is Counterfeit');
      _speak(verdict);

      notifyListeners();
      return result;
    } catch (e) {
      _error = 'Authentication failed: $e';
      _isProcessing = false;
      notifyListeners();
      return null;
    }
  }

  /// Run inference on an arbitrary list of view paths (used for Batch Mode).
  /// This does not alter the provider's main state or history.
  Future<AuthenticationResult?> runBatchInference(List<String> paths) async {
    if (!_isModelLoaded) {
      await loadModel();
      if (!_isModelLoaded) return null;
    }
    try {
      return await _onnxService.runInference(paths);
    } catch (e) {
      debugPrint('Batch Inference Error: $e');
      return null;
    }
  }

  /// Generate occlusion sensitivity heatmaps.
  Future<OcclusionResult?> generateHeatmaps() async {
    if (!allViewsCaptured || !_isModelLoaded) return null;

    _isGeneratingHeatmap = true;
    _heatmapProgress = 0;
    _heatmapTotal = OnnxService.numViews *
        OnnxService.occlusionGridSize *
        OnnxService.occlusionGridSize;
    notifyListeners();

    try {
      final result = await _onnxService.runOcclusionSensitivity(
        _capturedViews,
        onProgress: (current, total) {
          _heatmapProgress = current;
          _heatmapTotal = total;
          notifyListeners();
        },
      );
      _lastOcclusion = result;
      _isGeneratingHeatmap = false;
      notifyListeners();
      return result;
    } catch (e) {
      _error = 'Heatmap generation failed: $e';
      _isGeneratingHeatmap = false;
      notifyListeners();
      return null;
    }
  }

  // --- HISTORY ---

  Future<void> _loadHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getStringList('scan_history') ?? [];
    _history.clear();
    for (final json in data) {
      try {
        _history.add(ScanHistoryEntry.fromJson(
          jsonDecode(json) as Map<String, dynamic>,
        ));
      } catch (_) {}
    }
  }

  Future<void> _addToHistory(AuthenticationResult result) async {
    final entry = ScanHistoryEntry(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      result: result,
      viewPaths: List<String>.from(_capturedViews),
    );
    _history.insert(0, entry);
    // Keep max 50 entries
    if (_history.length > 50) _history.removeLast();
    await _saveHistory();
  }

  Future<void> _saveHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final data = _history.map((e) => jsonEncode(e.toJson())).toList();
    await prefs.setStringList('scan_history', data);
  }

  Future<void> deleteHistoryEntry(String id) async {
    _history.removeWhere((e) => e.id == id);
    await _saveHistory();
    notifyListeners();
  }

  Future<void> clearHistory() async {
    _history.clear();
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('scan_history');
    notifyListeners();
  }

  void reset() {
    _capturedViews.clear();
    _currentViewIndex = 0;
    _lastResult = null;
    _lastOcclusion = null;
    _error = null;
    _speak(currentInstruction);
    notifyListeners();
  }

  @override
  void dispose() {
    _onnxService.dispose();
    _tts.stop();
    super.dispose();
  }
}
