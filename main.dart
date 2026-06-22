import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const TrafficApp());
}

class TrafficApp extends StatelessWidget {
  const TrafficApp({super.key});

  @override
  Widget build(BuildContext context) {
    const seedColor = Color(0xFF0F766E);

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Traffic AI',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: seedColor,
          brightness: Brightness.light,
        ),
        scaffoldBackgroundColor: const Color(0xFFF4F7FA),
        appBarTheme: const AppBarTheme(
          centerTitle: false,
          elevation: 0,
          backgroundColor: Colors.transparent,
          foregroundColor: Color(0xFF0F172A),
        ),
        cardTheme: CardTheme(
          elevation: 10,
          shadowColor: Colors.black.withValues(alpha: 0.12),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
        ),
      ),
      home: const TrafficHomePage(),
    );
  }
}

class RoutePoint {
  const RoutePoint({required this.latitude, required this.longitude});

  final double latitude;
  final double longitude;

  LatLng get latLng => LatLng(latitude, longitude);
}

class TrafficPredictionResult {
  const TrafficPredictionResult({
    required this.estimatedMinutes,
    required this.distanceKm,
    required this.confidenceScore,
  });

  final double estimatedMinutes;
  final double distanceKm;
  final double confidenceScore;

  factory TrafficPredictionResult.fromJson(Map<String, dynamic> json) {
    final data = json['data'] as Map<String, dynamic>;
    return TrafficPredictionResult(
      estimatedMinutes: (data['estimated_minutes'] as num).toDouble(),
      distanceKm: (data['distance_km'] as num).toDouble(),
      confidenceScore: (data['confidence_score'] as num).toDouble(),
    );
  }
}

class TrafficPredictionRequest {
  const TrafficPredictionRequest({
    required this.origin,
    required this.destination,
    required this.dateTime,
  });

  final RoutePoint origin;
  final RoutePoint destination;
  final DateTime dateTime;

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'origin_latitude': origin.latitude,
      'origin_longitude': origin.longitude,
      'destination_latitude': destination.latitude,
      'destination_longitude': destination.longitude,
      'request_datetime': dateTime.toUtc().toIso8601String(),
    };
  }
}

class TrafficApiService {
  TrafficApiService({required this.baseUrl, http.Client? client})
      : _client = client ?? http.Client();

  final String baseUrl;
  final http.Client _client;

  Future<TrafficPredictionResult> predictDuration(
    TrafficPredictionRequest request,
  ) async {
    final uri = Uri.parse('$baseUrl/api/v1/predict-duration');
    final response = await _client
        .post(
          uri,
          headers: <String, String>{
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          body: jsonEncode(request.toJson()),
        )
        .timeout(const Duration(seconds: 20));

    if (response.statusCode < 200 || response.statusCode >= 300) {
      final message = _extractErrorMessage(response.body) ?? 'Sunucudan geçersiz yanıt alındı.';
      throw TrafficApiException(message);
    }

    final decoded = jsonDecode(response.body) as Map<String, dynamic>;
    return TrafficPredictionResult.fromJson(decoded);
  }

  String? _extractErrorMessage(String body) {
    try {
      final decoded = jsonDecode(body) as Map<String, dynamic>;
      final message = decoded['message'];
      if (message is String && message.isNotEmpty) {
        return message;
      }
    } catch (_) {
      return body.isNotEmpty ? body : null;
    }
    return null;
  }

  void dispose() {
    _client.close();
  }
}

class TrafficApiException implements Exception {
  TrafficApiException(this.message);

  final String message;

  @override
  String toString() => message;
}

class TrafficHomePage extends StatefulWidget {
  const TrafficHomePage({super.key});

  @override
  State<TrafficHomePage> createState() => _TrafficHomePageState();
}

class _TrafficHomePageState extends State<TrafficHomePage> {
  static const _initialCamera = CameraPosition(
    target: LatLng(39.9334, 32.8597),
    zoom: 6.2,
  );

  final TrafficApiService _apiService = TrafficApiService(baseUrl: 'http://10.0.2.2:8000');
  final DraggableScrollableController _sheetController = DraggableScrollableController();

  GoogleMapController? _mapController;
  RoutePoint? _origin;
  RoutePoint? _destination;
  DateTime _selectedDateTime = DateTime.now();
  bool _isLoading = false;
  TrafficPredictionResult? _prediction;

  @override
  void dispose() {
    _mapController?.dispose();
    _apiService.dispose();
    _sheetController.dispose();
    super.dispose();
  }

  Set<Marker> get _markers {
    final markers = <Marker>{};
    if (_origin != null) {
      markers.add(
        Marker(
          markerId: const MarkerId('origin'),
          position: _origin!.latLng,
          icon: BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueAzure),
          infoWindow: const InfoWindow(title: 'Başlangıç'),
        ),
      );
    }
    if (_destination != null) {
      markers.add(
        Marker(
          markerId: const MarkerId('destination'),
          position: _destination!.latLng,
          icon: BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueRed),
          infoWindow: const InfoWindow(title: 'Varış'),
        ),
      );
    }
    return markers;
  }

  void _onMapTapped(LatLng point) {
    setState(() {
      if (_origin == null || (_origin != null && _destination != null)) {
        _origin = RoutePoint(latitude: point.latitude, longitude: point.longitude);
        _destination = null;
        _prediction = null;
      } else {
        _destination = RoutePoint(latitude: point.latitude, longitude: point.longitude);
        _prediction = null;
      }
    });
    _fitMapToMarkers();
  }

  Future<void> _fitMapToMarkers() async {
    if (_mapController == null || _origin == null || _destination == null) {
      return;
    }
    final bounds = _createBounds(_origin!.latLng, _destination!.latLng);
    await Future<void>.delayed(const Duration(milliseconds: 60));
    if (!mounted) {
      return;
    }
    await _mapController!.animateCamera(
      CameraUpdate.newLatLngBounds(bounds, 80),
    );
  }

  LatLngBounds _createBounds(LatLng start, LatLng end) {
    final southWest = LatLng(
      start.latitude < end.latitude ? start.latitude : end.latitude,
      start.longitude < end.longitude ? start.longitude : end.longitude,
    );
    final northEast = LatLng(
      start.latitude > end.latitude ? start.latitude : end.latitude,
      start.longitude > end.longitude ? start.longitude : end.longitude,
    );
    return LatLngBounds(southwest: southWest, northeast: northEast);
  }

  Future<void> _pickDate() async {
    final date = await showDatePicker(
      context: context,
      initialDate: _selectedDateTime,
      firstDate: DateTime.now().subtract(const Duration(days: 1)),
      lastDate: DateTime.now().add(const Duration(days: 365)),
      helpText: 'Tarih Seçin',
    );
    if (date == null || !mounted) {
      return;
    }

    setState(() {
      _selectedDateTime = DateTime(
        date.year,
        date.month,
        date.day,
        _selectedDateTime.hour,
        _selectedDateTime.minute,
      );
    });
  }

  Future<void> _pickTime() async {
    final time = await showTimePicker(
      context: context,
      initialTime: TimeOfDay.fromDateTime(_selectedDateTime),
      helpText: 'Saat Seçin',
    );
    if (time == null || !mounted) {
      return;
    }

    setState(() {
      _selectedDateTime = DateTime(
        _selectedDateTime.year,
        _selectedDateTime.month,
        _selectedDateTime.day,
        time.hour,
        time.minute,
      );
    });
  }

  Future<void> _predict() async {
    final origin = _origin;
    final destination = _destination;
    if (origin == null || destination == null) {
      _showSnackBar('Lütfen başlangıç ve varış noktalarını seçin.');
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      final result = await _apiService.predictDuration(
        TrafficPredictionRequest(
          origin: origin,
          destination: destination,
          dateTime: _selectedDateTime,
        ),
      );
      if (!mounted) {
        return;
      }
      setState(() {
        _prediction = result;
      });
    } on TimeoutException {
      if (mounted) {
        _showSnackBar('İstek zaman aşımına uğradı. Lütfen tekrar deneyin.');
      }
    } on TrafficApiException catch (error) {
      if (mounted) {
        _showSnackBar(error.message);
      }
    } catch (_) {
      if (mounted) {
        _showSnackBar('Ağ bağlantısı sırasında bir hata oluştu.');
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context)
      ..clearSnackBars()
      ..showSnackBar(
        SnackBar(
          content: Text(message),
          behavior: SnackBarBehavior.floating,
          margin: const EdgeInsets.all(16),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        ),
      );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: Stack(
        children: <Widget>[
          Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: <Color>[Color(0xFFEAF4F3), Color(0xFFF7F9FC)],
              ),
            ),
          ),
          SafeArea(
            child: Column(
              children: <Widget>[
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 16, 20, 12),
                  child: Row(
                    children: <Widget>[
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: <Widget>[
                            Text(
                              'Akıllı Trafik Tahmini',
                              style: theme.textTheme.headlineSmall?.copyWith(
                                fontWeight: FontWeight.w800,
                                color: const Color(0xFF0F172A),
                              ),
                            ),
                            const SizedBox(height: 6),
                            Text(
                              'Başlangıç ve varış noktalarını haritadan seçin.',
                              style: theme.textTheme.bodyMedium?.copyWith(
                                color: const Color(0xFF475569),
                              ),
                            ),
                          ],
                        ),
                      ),
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.9),
                          borderRadius: BorderRadius.circular(18),
                          boxShadow: <BoxShadow>[
                            BoxShadow(
                              color: Colors.black.withValues(alpha: 0.08),
                              blurRadius: 24,
                              offset: const Offset(0, 10),
                            ),
                          ],
                        ),
                        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: <Widget>[
                            Text('Tarih', style: theme.textTheme.labelMedium),
                            TextButton(
                              onPressed: _pickDate,
                              child: Text(
                                '${_selectedDateTime.day.toString().padLeft(2, '0')}.${_selectedDateTime.month.toString().padLeft(2, '0')}.${_selectedDateTime.year}',
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(30),
                      child: GoogleMap(
                        initialCameraPosition: _initialCamera,
                        onMapCreated: (controller) {
                          _mapController = controller;
                        },
                        onTap: _onMapTapped,
                        markers: _markers,
                        myLocationButtonEnabled: false,
                        zoomControlsEnabled: false,
                        mapToolbarEnabled: false,
                        compassEnabled: false,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 12),
              ],
            ),
          ),
          DraggableScrollableSheet(
            controller: _sheetController,
            initialChildSize: 0.34,
            minChildSize: 0.26,
            maxChildSize: 0.74,
            builder: (BuildContext context, ScrollController scrollController) {
              return Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: const BorderRadius.vertical(top: Radius.circular(32)),
                  boxShadow: <BoxShadow>[
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.12),
                      blurRadius: 30,
                      offset: const Offset(0, -8),
                    ),
                  ],
                ),
                child: SingleChildScrollView(
                  controller: scrollController,
                  padding: const EdgeInsets.fromLTRB(20, 10, 20, 28),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: <Widget>[
                      Center(
                        child: Container(
                          width: 52,
                          height: 5,
                          decoration: BoxDecoration(
                            color: const Color(0xFFD7DFE8),
                            borderRadius: BorderRadius.circular(999),
                          ),
                        ),
                      ),
                      const SizedBox(height: 18),
                      Row(
                        children: <Widget>[
                          Expanded(
                            child: _SelectionCard(
                              title: 'Başlangıç',
                              value: _origin == null
                                  ? 'Haritadan seçin'
                                  : '${_origin!.latitude.toStringAsFixed(4)}, ${_origin!.longitude.toStringAsFixed(4)}',
                              accentColor: const Color(0xFF0284C7),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: _SelectionCard(
                              title: 'Varış',
                              value: _destination == null
                                  ? 'Haritadan seçin'
                                  : '${_destination!.latitude.toStringAsFixed(4)}, ${_destination!.longitude.toStringAsFixed(4)}',
                              accentColor: const Color(0xFFEF4444),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      Row(
                        children: <Widget>[
                          Expanded(
                            child: ElevatedButton.icon(
                              onPressed: _pickDate,
                              icon: const Icon(Icons.calendar_month_outlined),
                              label: const Text('Tarih'),
                              style: ElevatedButton.styleFrom(
                                padding: const EdgeInsets.symmetric(vertical: 14),
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                              ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: ElevatedButton.icon(
                              onPressed: _pickTime,
                              icon: const Icon(Icons.schedule_outlined),
                              label: const Text('Saat'),
                              style: ElevatedButton.styleFrom(
                                padding: const EdgeInsets.symmetric(vertical: 14),
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 14),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: const Color(0xFFF8FAFC),
                          borderRadius: BorderRadius.circular(24),
                          border: Border.all(color: const Color(0xFFE2E8F0)),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: <Widget>[
                            Text(
                              'Seçilen Zaman',
                              style: theme.textTheme.labelLarge?.copyWith(
                                color: const Color(0xFF475569),
                              ),
                            ),
                            const SizedBox(height: 6),
                            Text(
                              '${_selectedDateTime.day.toString().padLeft(2, '0')}.${_selectedDateTime.month.toString().padLeft(2, '0')}.${_selectedDateTime.year}  ${_selectedDateTime.hour.toString().padLeft(2, '0')}:${_selectedDateTime.minute.toString().padLeft(2, '0')}',
                              style: theme.textTheme.titleMedium?.copyWith(
                                fontWeight: FontWeight.w700,
                                color: const Color(0xFF0F172A),
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 16),
                      SizedBox(
                        width: double.infinity,
                        child: FilledButton(
                          onPressed: _isLoading ? null : _predict,
                          style: FilledButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 16),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                          ),
                          child: _isLoading
                              ? const SizedBox(
                                  height: 22,
                                  width: 22,
                                  child: CircularProgressIndicator(strokeWidth: 2.6),
                                )
                              : const Text('Süreyi Tahmin Et'),
                        ),
                      ),
                      const SizedBox(height: 16),
                      if (_prediction != null)
                        _PredictionPanel(result: _prediction!),
                    ],
                  ),
                ),
              );
            },
          ),
          if (_isLoading)
            Positioned.fill(
              child: Container(
                color: Colors.black.withValues(alpha: 0.12),
                child: const Center(
                  child: SizedBox(
                    height: 56,
                    width: 56,
                    child: CircularProgressIndicator(strokeWidth: 4),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _SelectionCard extends StatelessWidget {
  const _SelectionCard({
    required this.title,
    required this.value,
    required this.accentColor,
  });

  final String title;
  final String value;
  final Color accentColor;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: accentColor.withValues(alpha: 0.16)),
        boxShadow: <BoxShadow>[
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 18,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            title,
            style: theme.textTheme.labelLarge?.copyWith(
              color: accentColor,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            value,
            style: theme.textTheme.bodyMedium?.copyWith(
              fontWeight: FontWeight.w600,
              color: const Color(0xFF0F172A),
            ),
          ),
        ],
      ),
    );
  }
}

class _PredictionPanel extends StatelessWidget {
  const _PredictionPanel({required this.result});

  final TrafficPredictionResult result;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: <Color>[Color(0xFF0F766E), Color(0xFF115E59)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(24),
        boxShadow: <BoxShadow>[
          BoxShadow(
            color: const Color(0xFF0F766E).withValues(alpha: 0.28),
            blurRadius: 22,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            'Tahmin Sonucu',
            style: theme.textTheme.titleLarge?.copyWith(
              color: Colors.white,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 14),
          Row(
            children: <Widget>[
              Expanded(
                child: _MetricTile(
                  label: 'Süre',
                  value: '${result.estimatedMinutes.toStringAsFixed(1)} dk',
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _MetricTile(
                  label: 'Mesafe',
                  value: '${result.distanceKm.toStringAsFixed(2)} km',
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _MetricTile(
            label: 'Güven Skoru',
            value: '%${(result.confidenceScore * 100).toStringAsFixed(1)}',
            fullWidth: true,
          ),
        ],
      ),
    );
  }
}

class _MetricTile extends StatelessWidget {
  const _MetricTile({
    required this.label,
    required this.value,
    this.fullWidth = false,
  });

  final String label;
  final String value;
  final bool fullWidth;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: fullWidth ? double.infinity : null,
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            label,
            style: Theme.of(context).textTheme.labelLarge?.copyWith(
                  color: Colors.white.withValues(alpha: 0.82),
                ),
          ),
          const SizedBox(height: 6),
          Text(
            value,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  color: Colors.white,
                  fontWeight: FontWeight.w800,
                ),
          ),
        ],
      ),
    );
  }
}