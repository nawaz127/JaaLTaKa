import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:screenshot/screenshot.dart';
import 'package:share_plus/share_plus.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import '../models/auth_result.dart';
import '../providers/auth_provider.dart';
import '../services/pdf_service.dart';
import 'explanation_screen.dart';

/// Screen 3: Authentication result display with share feature.
/// Developed by Shah Nawaz.
class ResultScreen extends StatefulWidget {
  final AuthenticationResult result;

  const ResultScreen({super.key, required this.result});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen>
    with SingleTickerProviderStateMixin {
  final ScreenshotController _screenshotController = ScreenshotController();
  late AnimationController _animController;
  late Animation<double> _scaleAnim;

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );
    _scaleAnim = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _animController, curve: Curves.elasticOut),
    );
    _animController.forward();
  }

  @override
  void dispose() {
    _animController.dispose();
    super.dispose();
  }

  Future<void> _shareResult() async {
    final image = await _screenshotController.capture();
    if (image == null) return;

    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/jaaltaka_result.png');
    await file.writeAsBytes(image);

    await Share.shareXFiles(
      [XFile(file.path)],
      text: 'JaalTaka Result: ${widget.result.predictionLabel} '
          '(${widget.result.confidencePercent} confidence)\n'
          'Developed by Shah Nawaz',
    );
  }

  Future<void> _downloadPdf() async {
    try {
      await PdfService.generateAndSharePdf(widget.result);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to generate PDF: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final result = widget.result;
    final isAuth = result.isAuthentic;
    final mainColor = isAuth ? Colors.green : Colors.red;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Result'),
        actions: [
          IconButton(
            icon: const Icon(Icons.picture_as_pdf),
            onPressed: _downloadPdf,
            tooltip: 'Download PDF Report',
          ),
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: _shareResult,
            tooltip: 'Share screenshot',
          ),
        ],
      ),
      body: Screenshot(
        controller: _screenshotController,
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Main result card with animation
              ScaleTransition(
                scale: _scaleAnim,
                child: Card(
                  elevation: 4,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20),
                    side: BorderSide(color: mainColor, width: 2),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      children: [
                        Container(
                          width: 80,
                          height: 80,
                          decoration: BoxDecoration(
                            color: mainColor.withValues(alpha: 0.1),
                            shape: BoxShape.circle,
                          ),
                          child: Icon(
                            isAuth ? Icons.verified : Icons.dangerous,
                            color: mainColor,
                            size: 48,
                          ),
                        ),
                        const SizedBox(height: 16),
                        Text(
                          result.predictionLabel,
                          style: TextStyle(
                            fontSize: 28,
                            fontWeight: FontWeight.bold,
                            color: mainColor,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Confidence: ${result.confidencePercent}',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 16),
                        // Animated confidence bar
                        TweenAnimationBuilder<double>(
                          tween: Tween(begin: 0, end: result.confidence),
                          duration: const Duration(milliseconds: 1200),
                          curve: Curves.easeOutCubic,
                          builder: (_, value, __) {
                            return LinearProgressIndicator(
                              value: value,
                              backgroundColor: Colors.grey.shade200,
                              color: mainColor,
                              minHeight: 10,
                              borderRadius: BorderRadius.circular(5),
                            );
                          },
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // Class probabilities
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Classification Scores',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: 12),
                      ...result.classProbabilities.entries.map((entry) {
                        final color =
                            entry.key == 'Real' ? Colors.green : Colors.red;
                        return Padding(
                          padding: const EdgeInsets.symmetric(vertical: 4),
                          child: Row(
                            children: [
                              SizedBox(
                                width: 60,
                                child: Text(entry.key,
                                    style: const TextStyle(
                                        fontWeight: FontWeight.w500)),
                              ),
                              Expanded(
                                child: TweenAnimationBuilder<double>(
                                  tween: Tween(begin: 0, end: entry.value),
                                  duration:
                                      const Duration(milliseconds: 1000),
                                  builder: (_, val, __) {
                                    return LinearProgressIndicator(
                                      value: val,
                                      color: color,
                                      backgroundColor:
                                          color.withValues(alpha: 0.1),
                                      minHeight: 12,
                                      borderRadius: BorderRadius.circular(6),
                                    );
                                  },
                                ),
                              ),
                              const SizedBox(width: 8),
                              Text(
                                '${(entry.value * 100).toStringAsFixed(1)}%',
                                style: const TextStyle(
                                    fontWeight: FontWeight.w600),
                              ),
                            ],
                          ),
                        );
                      }),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // Inference time and Serial Number
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    children: [
                      Row(
                        children: [
                          const Icon(Icons.numbers, color: Colors.purple),
                          const SizedBox(width: 8),
                          Text(
                            'Serial Number: ${result.serialNumber}',
                            style: Theme.of(context).textTheme.bodyLarge,
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Row(
                        children: [
                          const Icon(Icons.speed, color: Colors.blue),
                          const SizedBox(width: 8),
                          Text(
                            'Inference: ${result.inferenceTimeMs.toStringAsFixed(0)} ms',
                            style: Theme.of(context).textTheme.bodyLarge,
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 8),

              // Developer credit
              Card(
                color: Colors.amber.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.code, color: Colors.amber.shade700, size: 16),
                      const SizedBox(width: 6),
                      Text(
                        'Developed by Shah Nawaz',
                        style: TextStyle(
                          color: Colors.amber.shade800,
                          fontWeight: FontWeight.w600,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // View explanation button
              OutlinedButton.icon(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) => ExplanationScreen(result: result),
                    ),
                  );
                },
                icon: const Icon(Icons.visibility),
                label: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 12),
                  child: Text('VIEW XAI EXPLANATION'),
                ),
              ),
              const SizedBox(height: 8),

              // New scan button
              FilledButton.icon(
                onPressed: () {
                  context.read<AuthenticationProvider>().reset();
                  Navigator.of(context).popUntil((route) => route.isFirst);
                },
                icon: const Icon(Icons.camera_alt),
                label: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 12),
                  child: Text('NEW SCAN'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
