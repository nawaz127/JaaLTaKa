import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/onnx_service.dart';
import '../models/auth_result.dart';

/// Manages authentication state, history, theme, and model inference.
/// Developed by Shah Nawaz.
class AuthenticationProvider extends ChangeNotifier {
  final OnnxService _onnxService = OnnxService();

  bool _isModelLoaded = false;
  bool _isProcessing = false;
  bool _isGeneratingHeatmap = false;
  AuthenticationResult? _lastResult;
  OcclusionResult? _lastOcclusion;
  String? _error;
  double _modelLoadProgress = 0.0;
  int _heatmapProgress = 0;
  int _heatmapTotal = 0;

  // Theme
  ThemeMode _themeMode = ThemeMode.system;

  // Multi-view capture state
  final List<String> _capturedViews = [];
  int _currentViewIndex = 0;

  // History
  final List<ScanHistoryEntry> _history = [];

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
  List<String> get capturedViews => List.unmodifiable(_capturedViews);
  int get currentViewIndex => _currentViewIndex;
  bool get allViewsCaptured => _capturedViews.length >= OnnxService.numViews;
  String get currentViewName =>
      _currentViewIndex < OnnxService.viewNames.length
          ? OnnxService.viewNames[_currentViewIndex]
          : '';
  List<ScanHistoryEntry> get history => List.unmodifiable(_history);

  /// Initialize provider — load theme preference and history.
  Future<void> initialize() async {
    final prefs = await SharedPreferences.getInstance();
    final themeIdx = prefs.getInt('themeMode') ?? 0;
    _themeMode = ThemeMode.values[themeIdx];
    await _loadHistory();
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
      _modelLoadProgress = 0.1;
      notifyListeners();
      _onnxService.init();
      _modelLoadProgress = 0.3;
      notifyListeners();
      await _onnxService.loadModel();
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
      notifyListeners();
    }
  }

  void removeLastView() {
    if (_capturedViews.isNotEmpty) {
      _capturedViews.removeLast();
      _currentViewIndex = _capturedViews.length;
      notifyListeners();
    }
  }

  /// Authenticate using all 6 captured views.
  Future<AuthenticationResult?> authenticateViews() async {
    if (!allViewsCaptured) {
      _error = 'Please capture all ${OnnxService.numViews} views first.';
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
    notifyListeners();
  }

  @override
  void dispose() {
    _onnxService.dispose();
    super.dispose();
  }
}
