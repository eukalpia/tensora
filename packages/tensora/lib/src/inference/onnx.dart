import '../errors/tensora_exception.dart';
import '../native/finalizer_callbacks.dart';
import '../native/native_inference_runtime.dart';
import '../native/native_runtime.dart';
import '../tensor/native_adoption.dart';
import '../tensor/tensor.dart';

/// A supported ONNX Runtime execution-provider preference.
enum OnnxExecutionProvider {
  /// Selects the best supported provider for the current platform and linked
  /// ONNX Runtime distribution.
  auto('auto'),

  /// Portable CPU execution.
  cpu('CPUExecutionProvider'),

  /// NVIDIA CUDA execution.
  cuda('CUDAExecutionProvider'),

  /// DirectML execution on a DirectX 12 adapter on Windows.
  directML('DmlExecutionProvider'),

  /// Apple Core ML execution on macOS.
  coreML('CoreMLExecutionProvider'),

  /// Intel OpenVINO execution.
  openVino('OpenVINOExecutionProvider'),

  /// AMD MIGraphX execution.
  miGraphX('MIGraphXExecutionProvider');

  const OnnxExecutionProvider(this.runtimeName);

  /// ONNX Runtime execution-provider name, or `auto` for automatic selection.
  final String runtimeName;

  static OnnxExecutionProvider fromRuntimeName(String name) {
    for (final provider in values) {
      if (provider.runtimeName == name) return provider;
    }
    throw NativeRuntimeException(
      'Native runtime returned unknown ONNX provider "$name".',
      operation: 'onnx.session.provider',
    );
  }
}

final Finalizer<int> _onnxSessionFinalizer = Finalizer<int>(
  releaseOnnxSessionFromFinalizer,
);

/// ONNX Runtime capability and provider diagnostics.
abstract final class OnnxRuntime {
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
    required this.requestedProvider,
    required this.selectedProvider,
    required this.profilingEnabled,
  }) {
    _onnxSessionFinalizer.attach(this, _handle, detach: this);
  }

  /// Loads an ONNX model from [modelPath].
  ///
  /// An explicit [provider] never silently falls back to CPU. Use
  /// [OnnxExecutionProvider.auto] when controlled platform-aware fallback is
  /// desired.
  factory OnnxSession(
    String modelPath, {
    OnnxExecutionProvider provider = OnnxExecutionProvider.auto,
    bool enableProfiling = false,
    String? profilingPrefix,
  }) {
    final runtime = NativeInferenceRuntime.instance;
    final handle = runtime.createSession(
      modelPath,
      providerName: provider.runtimeName,
      enableProfiling: enableProfiling,
      profilingPrefix: profilingPrefix,
    );
    try {
      final selected = OnnxExecutionProvider.fromRuntimeName(
        runtime.sessionProvider(handle),
      );
      if (provider != OnnxExecutionProvider.auto && selected != provider) {
        throw NativeRuntimeException(
          'Requested ONNX provider ${provider.runtimeName}, but native runtime '
          'selected ${selected.runtimeName}.',
          operation: 'onnx.session.create',
        );
      }
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
        requestedProvider: provider,
        selectedProvider: selected,
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

  /// Provider preference supplied when the session was created.
  final OnnxExecutionProvider requestedProvider;

  /// Provider actually attached to this native session.
  final OnnxExecutionProvider selectedProvider;

  /// Whether this session was created with native profiling enabled.
  final bool profilingEnabled;

  /// Whether deterministic session release has completed.
  bool get isDisposed => _disposed;

  /// Executes inference using named Tensora tensors.
  ///
  /// The portable inference contract accepts dense float32 tensors. Outputs
  /// default to every model output and are returned in model/requested order.
  Map<String, Tensor> run(Map<String, Tensor> inputs, {List<String>? outputs}) {
    _ensureLive('run');
    _validateInputs(inputs);
    final requestedOutputs =
        outputs == null ? outputNames : List<String>.unmodifiable(outputs);
    _validateOutputs(requestedOutputs);

    final orderedInputs = <Tensor>[];
    for (final name in inputNames) {
      orderedInputs.add(inputs[name]!);
    }
    final inputHandles = orderedInputs
        .map(
          (tensor) => tensor.nativeHandleForRuntime(nativeTensorAdoptionToken),
        )
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
          Tensor.adoptNativeHandleForRuntime(handle, nativeTensorAdoptionToken),
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
      for (var index = adopted.length + 1; index < handles.length; index++) {
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
    for (final supplied in inputs.keys) {
      if (!inputNames.contains(supplied)) {
        throw InvalidArgumentException(
          'Unknown ONNX input "$supplied".',
          operation: 'onnx.session.run',
        );
      }
    }
    for (final expected in inputNames) {
      if (!inputs.containsKey(expected)) {
        throw InvalidArgumentException(
          'Missing ONNX input "$expected".',
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
