import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart' as mt;
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';
import 'package:intl/intl.dart';
import '../models/auth_result.dart';

class PdfService {
  /// Generates a PDF report for the authentication result and opens the share/print dialog.
  static Future<void> generateAndSharePdf(AuthenticationResult result) async {
    final pdf = pw.Document();

    // Load images
    final List<pw.MemoryImage> viewImages = [];
    for (var view in result.viewResults) {
      try {
        final file = File(view.imagePath);
        if (await file.exists()) {
          final bytes = await file.readAsBytes();
          viewImages.add(pw.MemoryImage(bytes));
        }
      } catch (e) {
        debugPrint('Error loading image for PDF: $e');
      }
    }

    final isAuth = result.isAuthentic;
    final primaryColor = isAuth ? PdfColors.green700 : PdfColors.red700;
    final dateStr = DateFormat('yyyy-MM-dd HH:mm:ss').format(result.timestamp);

    pdf.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.a4,
        margin: const pw.EdgeInsets.all(32),
        build: (pw.Context context) {
          return [
            _buildHeader(dateStr),
            pw.SizedBox(height: 20),
            _buildVerdict(result, primaryColor),
            pw.SizedBox(height: 20),
            _buildMetrics(result),
            pw.SizedBox(height: 20),
            _buildProbabilities(result),
            pw.SizedBox(height: 20),
            _buildAttentionWeights(result),
            pw.SizedBox(height: 20),
            pw.Text(
              'Captured Views:',
              style: pw.TextStyle(fontSize: 16, fontWeight: pw.FontWeight.bold),
            ),
            pw.SizedBox(height: 10),
            _buildImagesGrid(result, viewImages),
          ];
        },
        footer: (pw.Context context) {
          return pw.Container(
            alignment: pw.Alignment.centerRight,
            margin: const pw.EdgeInsets.only(top: 10),
            child: pw.Text(
              'Page ${context.pageNumber} of ${context.pagesCount}',
              style: pw.TextStyle(color: PdfColors.grey),
            ),
          );
        },
      ),
    );

    // Share or print the PDF
    await Printing.sharePdf(
      bytes: await pdf.save(),
      filename: 'JaalTaka_Report_${DateFormat('yyyyMMdd_HHmmss').format(result.timestamp)}.pdf',
    );
  }

  static pw.Widget _buildHeader(String dateStr) {
    return pw.Column(
      crossAxisAlignment: pw.CrossAxisAlignment.center,
      children: [
        pw.Text(
          'JaalTaka Authentication Report',
          style: pw.TextStyle(fontSize: 24, fontWeight: pw.FontWeight.bold),
        ),
        pw.SizedBox(height: 5),
        pw.Text(
          'Generated: $dateStr',
          style: const pw.TextStyle(fontSize: 12),
        ),
        pw.Text(
          'Developed by Shah Nawaz',
          style: pw.TextStyle(fontSize: 12, color: PdfColors.blue800),
        ),
        pw.Divider(),
      ],
    );
  }

  static pw.Widget _buildVerdict(AuthenticationResult result, PdfColor color) {
    return pw.Container(
      padding: const pw.EdgeInsets.all(15),
      decoration: pw.BoxDecoration(
        border: pw.Border.all(color: color, width: 2),
        borderRadius: const pw.BorderRadius.all(pw.Radius.circular(10)),
        color: result.isAuthentic ? PdfColors.green50 : PdfColors.red50,
      ),
      child: pw.Center(
        child: pw.Text(
          'Verdict: ${result.predictionLabel}',
          style: pw.TextStyle(
            fontSize: 22,
            fontWeight: pw.FontWeight.bold,
            color: color,
          ),
        ),
      ),
    );
  }

  static pw.Widget _buildMetrics(AuthenticationResult result) {
    return pw.Row(
      mainAxisAlignment: pw.MainAxisAlignment.spaceAround,
      children: [
        _buildMetricItem('Confidence', result.confidencePercent),
        _buildMetricItem('Inference Time', '${result.inferenceTimeMs.toStringAsFixed(0)} ms'),
        _buildMetricItem('Engine', 'ONNX (INT8)'),
      ],
    );
  }

  static pw.Widget _buildMetricItem(String label, String value) {
    return pw.Column(
      children: [
        pw.Text(label, style: pw.TextStyle(color: PdfColors.grey700)),
        pw.SizedBox(height: 5),
        pw.Text(value, style: pw.TextStyle(fontSize: 16, fontWeight: pw.FontWeight.bold)),
      ],
    );
  }

  static pw.Widget _buildProbabilities(AuthenticationResult result) {
    return pw.Column(
      crossAxisAlignment: pw.CrossAxisAlignment.start,
      children: [
        pw.Text('Class Probabilities:', style: pw.TextStyle(fontSize: 16, fontWeight: pw.FontWeight.bold)),
        pw.SizedBox(height: 8),
        ...result.classProbabilities.entries.map((e) {
          return pw.Padding(
            padding: const pw.EdgeInsets.only(bottom: 4),
            child: pw.Row(
              children: [
                pw.SizedBox(width: 80, child: pw.Text(e.key)),
                pw.Text(': ${(e.value * 100).toStringAsFixed(2)}%'),
              ],
            ),
          );
        }),
      ],
    );
  }

  static pw.Widget _buildAttentionWeights(AuthenticationResult result) {
    return pw.Column(
      crossAxisAlignment: pw.CrossAxisAlignment.start,
      children: [
        pw.Text('Attention / SHAP Weights:', style: pw.TextStyle(fontSize: 16, fontWeight: pw.FontWeight.bold)),
        pw.SizedBox(height: 8),
        ...result.viewResults.map((v) {
          return pw.Padding(
            padding: const pw.EdgeInsets.only(bottom: 4),
            child: pw.Row(
              children: [
                pw.SizedBox(width: 120, child: pw.Text(v.name)),
                pw.Text(': ${(v.importance).toStringAsFixed(4)}'),
              ],
            ),
          );
        }),
      ],
    );
  }

  static pw.Widget _buildImagesGrid(AuthenticationResult result, List<pw.MemoryImage> viewImages) {
    if (viewImages.isEmpty) {
      return pw.Text('No images available.');
    }

    final rows = <pw.Widget>[];
    for (var i = 0; i < viewImages.length; i += 2) {
      final cells = <pw.Widget>[];
      
      // First image in row
      cells.add(_buildImageCell(result.viewResults[i].name, viewImages[i]));
      
      // Second image in row (if exists)
      if (i + 1 < viewImages.length) {
        cells.add(pw.SizedBox(width: 20));
        cells.add(_buildImageCell(result.viewResults[i + 1].name, viewImages[i + 1]));
      }
      
      rows.add(
        pw.Row(
          mainAxisAlignment: pw.MainAxisAlignment.start,
          crossAxisAlignment: pw.CrossAxisAlignment.start,
          children: cells,
        ),
      );
      rows.add(pw.SizedBox(height: 20));
    }

    return pw.Column(children: rows);
  }

  static pw.Widget _buildImageCell(String name, pw.MemoryImage image) {
    return pw.Expanded(
      child: pw.Column(
        crossAxisAlignment: pw.CrossAxisAlignment.center,
        children: [
          pw.Text(name, style: pw.TextStyle(fontWeight: pw.FontWeight.bold)),
          pw.SizedBox(height: 5),
          pw.Container(
            height: 120,
            decoration: pw.BoxDecoration(
              border: pw.Border.all(color: PdfColors.grey400),
            ),
            child: pw.Image(image, fit: pw.BoxFit.contain),
          ),
        ],
      ),
    );
  }
}
