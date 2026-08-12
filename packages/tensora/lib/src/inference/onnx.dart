import '../errors/tensora_exception.dart';
import '../native/native_inference_runtime.dart';
import '../native/native_runtime.dart';
import '../tensor/native_adoption.dart';
import '../tensor/tensor.dart';

final Finalizer<int> _onnxSessionFinalizer = Finalizer<int>((handle) {
  NativeInferenceRuntime.instance.releaseFromFinalizer(handle);
});

/// ONNX Runtime capability and provider diagnostics.
final class OnnxRuntime {
  OnnxRuntime._();

  /// Whether the loaded native library contains the ONNX Runtime backend.
  static bool get available => NativeInferenceRuntime.instance.available();

  /// Execution providers available in the linked ONNX Runtime distribution.
  static List<String> get providers =>
      List<String>.unmodifiable(NativeInferenceRuntime.instance.providers());

  /// Number of currently live native ONNX session handles.
  static int get liveSessionCount =>
      NativeInferenceRuntime.instance.liveSessionCount();
}

/// A reusable native ONNX inference session.
@pragma('vm:isolate-unsendable')
final class OnnxSession {
  OnnxSession._(
    this._handle, {
    required this.inputNames,
    required this.outputNames,
    required this.profilingEnabled,
  }) {
    _onnxSessionFinalizer.attach(this, _handle, detach: this);
  }

  /// Loads an ONNX model from [modelPath].
  factory OnnxSession(
    String modelPath, {
    bool enableProfiling = false,
    String? profilingPrefix,
  }) {
    final runtime = NativeInferenceRuntime.instance;
    final handle = runtime.createSession(
      modelPath,
      enableProfiling: enableProfiling,
      profilingPrefix: profilingPrefix,
    );
    try {
      final inputs = runtime.inputNames(handle);
      final outputs = runtime.outputNames(handle);
      if (inputs.isEmpty) {
        throw ModelRuntimeException(
          'ONNX model exposes no inputs.',
          operation: 'onnx.session.create',
        );
      }
      if (outputs.isEmpty) {
        throw ModelRuntimeException(
          'ONNX model exposes no outputs.',
          operation: 'onnx.session.create',
        );
      }
      return OnnxSession._(
        handle,
        inputNames: List<String>.unmodifiable(inputs),
        outputNames: List<String>.unmodifiable(outputs),
        profilingEnabled: enableProfiling,
      );
    } catch (_) {
      runtime.releaseFromFinalizer(handle);
      rethrow;
    }
  }

  int _handle;
  bool _disposed = false;
  bool _profilingEnded = false;

  /// Model input names in native model order.
  final List<String> inputNames;

  /// Model output names in native model order.
  final List<String> outputNames;

  /// Whether this session was created with native profiling enabled.
  final bool profilingEnabled;

  /// Whether deterministic session release has completed.
  bool get isDisposed => _disposed;

  /// Executes inference using named Tensora tensors.
  ///
  /// The initial portable-inference contract accepts dense float32 tensors.
  /// Outputs default to every model output and are returned in model/requested
  /// order through the insertion-ordered Dart map.
  Map<String, Tensor> run(
    Map<String, Tensor> inputs, {
    List<String>? outputs,
  }) {
    _ensureLive('run');
    _validateInputs(inputs);
    final requestedOutputs = outputs == null
        ? outputNames
        : List<String>.unmodifiable(outputs);
    _validateOutputs(requestedOutputs);

    final orderedInputs = <Tensor>[];
    for (final name in inputNames) {
      orderedInputs.add(inputs[name]!);
    }
    final inputHandles = orderedInputs
        .map((tensor) => tensor.nativeHandleForRuntime(nativeTensorAdoptionToken))
        .toList(growable: false);

    final handles = NativeInferenceRuntime.instance.run(
      _handle,
      inputNames: inputNames,
      inputHandles: inputHandles,
      outputNames: requestedOutputs,
    );

    final adopted = <Tensor>[];
    try {
      for (final handle in handles) {
        adopted.add(
          Tensor.adoptNativeHandleForRuntime(
            handle,
            nativeTensorAdoptionToken,
          ),
        );
      }
      return <String, Tensor>{
        for (var index = 0; index < requestedOutputs.length; index++)
          requestedOutputs[index]: adopted[index],
      };
    } catch (_) {
      for (final tensor in adopted) {
        tensor.dispose();
      }
      for (var index = adopted.length; index < handles.length; index++) {
        NativeRuntime.instance.releaseFromFinalizer(handles[index]);
      }
      rethrow;
    }
  }

  /// Ends native profiling and returns the generated profiling file path.
  String endProfiling() {
    _ensureLive('endProfiling');
    if (!profilingEnabled) {
      throw InvalidArgumentException(
        'This ONNX session was not created with profiling enabled.',
        operation: 'onnx.session.endProfiling',
      );
    }
    if (_profilingEnded) {
      throw InvalidArgumentException(
        'Profiling has already ended for this ONNX session.',
        operation: 'onnx.session.endProfiling',
      );
    }
    final path = NativeInferenceRuntime.instance.endProfiling(_handle);
    _profilingEnded = true;
    return path;
  }

  /// Deterministically releases the native session handle.
  void dispose() {
    if (_disposed) return;
    NativeInferenceRuntime.instance.release(_handle);
    _onnxSessionFinalizer.detach(this);
    _handle = 0;
    _disposed = true;
  }

  void _validateInputs(Map<String, Tensor> inputs) {
    if (inputs.length != inputNames.length) {
      throw InvalidArgumentException(
        'Expected ${inputNames.length} ONNX inputs, got ${inputs.length}.',
        operation: 'onnx.session.run',
      );
    }
    for (final expected in inputNames) {
      if (!inputs.containsKey(expected)) {
        throw InvalidArgumentException(
          'Missing ONNX input "$expected".',
          operation: 'onnx.session.run',
        );
      }
    }
    for (final supplied in inputs.keys) {
      if (!inputNames.contains(supplied)) {
        throw InvalidArgumentException(
          'Unknown ONNX input "$supplied".',
          operation: 'onnx.session.run',
        );
      }
    }
  }

  void _validateOutputs(List<String> outputs) {
    if (outputs.isEmpty) {
      throw InvalidArgumentException(
        'At least one ONNX output must be requested.',
        operation: 'onnx.session.run',
      );
    }
    final seen = <String>{};
    for (final output in outputs) {
      if (!outputNames.contains(output)) {
        throw InvalidArgumentException(
          'Unknown ONNX output "$output".',
          operation: 'onnx.session.run',
        );
      }
      if (!seen.add(output)) {
        throw InvalidArgumentException(
          'ONNX output "$output" was requested more than once.',
          operation: 'onnx.session.run',
        );
      }
    }
  }

  void _ensureLive(String operation) {
    if (_disposed) {
      throw NativeRuntimeException(
        'ONNX session has already been disposed.',
        operation: 'onnx.session.$operation',
      );
    }
  }
}
