import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';
import '../models/auth_result.dart';
import '../services/onnx_service.dart';
import 'result_screen.dart';

class BatchScreen extends StatefulWidget {
  const BatchScreen({super.key});

  @override
  State<BatchScreen> createState() => _BatchScreenState();
}

class _BatchScreenState extends State<BatchScreen> {
  final ImagePicker _picker = ImagePicker();
  bool _isProcessing = false;
  List<AuthenticationResult> _results = [];
  double _progress = 0.0;

  Future<void> _pickAndProcessBatch() async {
    final images = await _picker.pickMultiImage();
    if (images.isEmpty) return;

    if (images.length % OnnxService.numViews != 0) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Please select images in multiples of ${OnnxService.numViews} (found ${images.length}).'),
            backgroundColor: Colors.red,
          ),
        );
      }
      return;
    }

    setState(() {
      _isProcessing = true;
      _results.clear();
      _progress = 0.0;
    });

    final provider = context.read<AuthenticationProvider>();
    final int totalNotes = images.length ~/ OnnxService.numViews;

    for (int i = 0; i < totalNotes; i++) {
      final startIndex = i * OnnxService.numViews;
      final noteImages = images.sublist(startIndex, startIndex + OnnxService.numViews);
      
      final viewPaths = noteImages.map((img) => img.path).toList();
      
      try {
        final result = await provider.runBatchInference(viewPaths);
        if (result != null) {
          _results.add(result);
        }
      } catch (e) {
        debugPrint('Error processing note $i: $e');
      }

      setState(() {
        _progress = (i + 1) / totalNotes;
      });
    }

    setState(() {
      _isProcessing = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    int authenticCount = _results.where((r) => r.isAuthentic).length;
    int fakeCount = _results.length - authenticCount;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Batch Analysis'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Select images from your gallery to process multiple banknotes at once. '
              'Images must be selected in exact multiples of 6 (Front, Back, Watermark, Thread, Serial, Hologram).',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: _isProcessing ? null : _pickAndProcessBatch,
              icon: const Icon(Icons.photo_library),
              label: const Padding(
                padding: EdgeInsets.symmetric(vertical: 12.0),
                child: Text('SELECT IMAGES FOR BATCH', style: TextStyle(fontSize: 16)),
              ),
            ),
            const SizedBox(height: 20),
            if (_isProcessing) ...[
              LinearProgressIndicator(value: _progress),
              const SizedBox(height: 10),
              Text(
                'Processing... ${(_progress * 100).toStringAsFixed(0)}%',
                textAlign: TextAlign.center,
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
            ],
            if (!_isProcessing && _results.isNotEmpty) ...[
              const Divider(),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _buildSummaryCard('Total', _results.length.toString(), Colors.blue),
                  _buildSummaryCard('Authentic', authenticCount.toString(), Colors.green),
                  _buildSummaryCard('Fake', fakeCount.toString(), Colors.red),
                ],
              ),
              const SizedBox(height: 10),
              Expanded(
                child: ListView.builder(
                  itemCount: _results.length,
                  itemBuilder: (context, index) {
                    final res = _results[index];
                    final isAuth = res.isAuthentic;
                    return Card(
                      child: ListTile(
                        leading: Icon(
                          isAuth ? Icons.verified : Icons.dangerous,
                          color: isAuth ? Colors.green : Colors.red,
                        ),
                        title: Text('Note ${index + 1}: ${res.predictionLabel}'),
                        subtitle: Text('Confidence: ${res.confidencePercent} | Time: ${res.inferenceTimeMs.toStringAsFixed(0)}ms'),
                        trailing: const Icon(Icons.chevron_right),
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => ResultScreen(result: res),
                            ),
                          );
                        },
                      ),
                    );
                  },
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildSummaryCard(String title, String value, Color color) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Text(value, style: TextStyle(fontSize: 24, color: color, fontWeight: FontWeight.bold)),
          ],
        ),
      ),
    );
  }
}
