import 'package:tensora/tensora.dart' as core;

import 'parameter.dart';
import 'state_dict.dart';

/// A named child module in a deterministic module tree.
final class NamedModule {
  const NamedModule(this.name, this.module);

  final String name;
  final Module module;
}

/// Base class for composable neural-network modules.
abstract base class Module {
  Module();

  Module? _owner;
  bool _disposed = false;
  bool _training = true;

  bool get isDisposed => _disposed;
  bool get isTraining => _training;

  /// Whether all lazily-created module structure is materialized.
  bool get isMaterialized => true;

  /// Public immutable snapshot of direct children.
  List<Module> get children {
    _ensureLive('children');
    internalEnsureMaterialized();
    return List<Module>.unmodifiable(
      internalRegisteredChildren.map((entry) => entry.module),
    );
  }

  /// Executes this module.
  core.Tensor call(core.Tensor input) {
    _ensureLive('forward');
    internalEnsureMaterialized();
    return forward(input);
  }

  /// Subclasses implement their forward computation here.
  core.Tensor forward(core.Tensor input);

  /// Deterministic unique parameter traversal.
  List<Parameter> get parameters => List<Parameter>.unmodifiable(
    namedParameters.map((entry) => entry.parameter),
  );

  /// Deterministic unique named parameter traversal.
  List<NamedParameter> get namedParameters {
    _ensureLive('namedParameters');
    internalEnsureMaterialized();
    final output = <NamedParameter>[];
    final seen = <int>{};
    _collectParameters('', output, seen);
    return List<NamedParameter>.unmodifiable(output);
  }

  /// Deterministic unique buffer traversal.
  List<Buffer> get buffers =>
      List<Buffer>.unmodifiable(namedBuffers.map((entry) => entry.buffer));

  /// Deterministic unique named buffer traversal.
  List<NamedBuffer> get namedBuffers {
    _ensureLive('namedBuffers');
    internalEnsureMaterialized();
    final output = <NamedBuffer>[];
    final seen = <int>{};
    _collectBuffers('', output, seen);
    return List<NamedBuffer>.unmodifiable(output);
  }

  /// This module followed by every descendant in deterministic tree order.
  List<Module> get modules =>
      List<Module>.unmodifiable(namedModules.map((entry) => entry.module));

  /// Named module traversal. The root has an empty path.
  List<NamedModule> get namedModules {
    _ensureLive('namedModules');
    internalEnsureMaterialized();
    final output = <NamedModule>[];
    final seen = <Module>{};
    _collectModules('', output, seen);
    return List<NamedModule>.unmodifiable(output);
  }

  /// Puts this full module tree into training mode.
  void train() => _setTraining(true);

  /// Puts this full module tree into evaluation mode.
  void eval() => _setTraining(false);

  /// Moves this full module tree to [device] after a capability preflight.
  void to(core.Device device) {
    _ensureLive('to');
    internalEnsureMaterialized();
    if (core.TensoraRuntime.deviceCount(device) <= 0) {
      throw core.UnsupportedOperationException(
        'Requested device is not available in the loaded training runtime.',
        operation: 'module.to',
      );
    }
    _preflightMove(device, <Module>{});
    _applyMove(device, <Module>{});
  }

  /// Captures parameters and persistent buffers without copying values to Dart.
  StateDict stateDict() {
    _ensureLive('stateDict');
    final snapshots = <String, core.Tensor>{};
    try {
      for (final entry in namedParameters) {
        snapshots[entry.name] = entry.parameter.snapshot();
      }
      for (final entry in namedBuffers) {
        if (entry.buffer.persistent) {
          snapshots[entry.name] = entry.buffer.snapshot();
        }
      }
      return StateDict.fromOwned(snapshots);
    } catch (_) {
      for (final tensor in snapshots.values) {
        tensor.dispose();
      }
      rethrow;
    }
  }

  /// Validates and transactionally restores an in-memory state snapshot.
  StateLoadResult loadStateDict(StateDict state, {bool strict = true}) {
    _ensureLive('loadStateDict');
    final targets = <String, Object>{};
    for (final entry in namedParameters) {
      targets[entry.name] = entry.parameter;
    }
    for (final entry in namedBuffers) {
      if (entry.buffer.persistent) targets[entry.name] = entry.buffer;
    }

    final sourceKeys = state.keys.toSet();
    final targetKeys = targets.keys.toSet();
    final missing = targetKeys.difference(sourceKeys).toList()..sort();
    final unexpected = sourceKeys.difference(targetKeys).toList()..sort();
    final result = StateLoadResult(
      missingKeys: missing,
      unexpectedKeys: unexpected,
    );

    if (strict && !result.isSuccess) {
      throw core.InvalidArgumentException(
        'StateDict keys are incompatible: missing=$missing, '
        'unexpected=$unexpected.',
        operation: 'module.loadStateDict',
      );
    }

    final common = targetKeys.intersection(sourceKeys).toList()..sort();
    final targetTensors = <core.Tensor>[];
    final sourceTensors = <core.Tensor>[];
    for (final key in common) {
      final source = state[key]!;
      final target = targets[key]!;
      final core.Shape shape;
      final core.DType dtype;
      final core.Device device;
      final core.Tensor targetTensor;
      if (target is Parameter) {
        shape = target.shape;
        dtype = target.dtype;
        device = target.device;
        targetTensor = target.tensorForRuntime;
      } else {
        final buffer = target as Buffer;
        shape = buffer.shape;
        dtype = buffer.dtype;
        device = buffer.device;
        targetTensor = buffer.tensorForRuntime;
      }
      if (source.shape != shape ||
          source.dtype != dtype ||
          source.device != device) {
        throw core.InvalidArgumentException(
          'StateDict entry $key has incompatible tensor metadata.',
          operation: 'module.loadStateDict',
        );
      }
      targetTensors.add(targetTensor);
      sourceTensors.add(source);
    }

    core.NativeTensorState.assignMany(
      targets: targetTensors,
      sources: sourceTensors,
    );
    return result;
  }

  /// Flutter-like tree representation without tensor host copies.
  String toTreeString() {
    _ensureLive('toTreeString');
    internalEnsureMaterialized();
    final buffer = StringBuffer()..writeln(internalDiagnosticLabel);
    _writeChildren(buffer, '', internalRegisteredChildren);
    return buffer.toString().trimRight();
  }

  /// Deterministically releases all resources owned by this module tree.
  void dispose() {
    if (_disposed) return;
    _disposed = true;

    final seenChildren = <Module>{};
    for (final entry in internalRegisteredChildren) {
      if (seenChildren.add(entry.module)) entry.module.dispose();
    }
    final seenParameters = <int>{};
    for (final entry in internalRegisteredParameters) {
      if (seenParameters.add(entry.parameter.identity)) {
        entry.parameter.dispose();
      }
    }
    final seenBuffers = <int>{};
    for (final entry in internalRegisteredBuffers) {
      if (seenBuffers.add(entry.buffer.identity)) entry.buffer.dispose();
    }
    internalOnDispose();
    _owner = null;
  }

  /// @nodoc
  List<NamedModule> get internalRegisteredChildren => const <NamedModule>[];

  /// @nodoc
  List<NamedParameter> get internalRegisteredParameters =>
      const <NamedParameter>[];

  /// @nodoc
  List<NamedBuffer> get internalRegisteredBuffers => const <NamedBuffer>[];

  /// @nodoc
  String get internalDiagnosticLabel => runtimeType.toString();

  /// @nodoc
  Module? get internalOwner => _owner;

  /// @nodoc
  void internalEnsureMaterialized() {}

  /// @nodoc
  void internalAttachOwner(Module owner) {
    _ensureLive('attachOwner');
    if (identical(owner, this)) {
      throw ArgumentError('A module cannot own itself.');
    }
    if (_owner != null) {
      throw ArgumentError(
        'Module $runtimeType already belongs to another owning path.',
      );
    }
    _owner = owner;
  }

  /// @nodoc
  void internalDetachOwner(Module owner) {
    if (identical(_owner, owner)) _owner = null;
  }

  /// @nodoc
  bool internalContainsModule(Module target, Set<Module> visited) {
    if (!visited.add(this)) return false;
    if (identical(this, target)) return true;
    internalEnsureMaterialized();
    for (final child in internalRegisteredChildren) {
      if (child.module.internalContainsModule(target, visited)) return true;
    }
    return false;
  }

  /// @nodoc
  void internalOnTrainingModeChanged(bool training) {}

  /// @nodoc
  void internalPreflightMove(core.Device device) {}

  /// @nodoc
  void internalOnMove(core.Device device) {}

  /// @nodoc
  void internalOnDispose() {}

  void _ensureLive(String operation) {
    if (_disposed) {
      throw core.DisposedTensorException(
        'Module has already been disposed.',
        operation: 'module.$operation',
      );
    }
  }

  void _setTraining(bool training) {
    _ensureLive(training ? 'train' : 'eval');
    internalEnsureMaterialized();
    _applyTraining(training, <Module>{});
  }

  void _applyTraining(bool training, Set<Module> visited) {
    if (!visited.add(this)) return;
    internalEnsureMaterialized();
    _training = training;
    internalOnTrainingModeChanged(training);
    for (final child in internalRegisteredChildren) {
      child.module._applyTraining(training, visited);
    }
  }

  void _preflightMove(core.Device device, Set<Module> visited) {
    if (!visited.add(this)) return;
    internalEnsureMaterialized();
    internalPreflightMove(device);
    for (final child in internalRegisteredChildren) {
      child.module._preflightMove(device, visited);
    }
  }

  void _applyMove(core.Device device, Set<Module> visited) {
    if (!visited.add(this)) return;
    internalOnMove(device);
    for (final child in internalRegisteredChildren) {
      child.module._applyMove(device, visited);
    }
  }

  void _collectParameters(
    String prefix,
    List<NamedParameter> output,
    Set<int> seen,
  ) {
    internalEnsureMaterialized();
    for (final entry in internalRegisteredParameters) {
      if (seen.add(entry.parameter.identity)) {
        output.add(
          NamedParameter(_joinPath(prefix, entry.name), entry.parameter),
        );
      }
    }
    for (final child in internalRegisteredChildren) {
      child.module._collectParameters(
        _joinPath(prefix, child.name),
        output,
        seen,
      );
    }
  }

  void _collectBuffers(String prefix, List<NamedBuffer> output, Set<int> seen) {
    internalEnsureMaterialized();
    for (final entry in internalRegisteredBuffers) {
      if (seen.add(entry.buffer.identity)) {
        output.add(NamedBuffer(_joinPath(prefix, entry.name), entry.buffer));
      }
    }
    for (final child in internalRegisteredChildren) {
      child.module._collectBuffers(_joinPath(prefix, child.name), output, seen);
    }
  }

  void _collectModules(
    String prefix,
    List<NamedModule> output,
    Set<Module> seen,
  ) {
    internalEnsureMaterialized();
    if (!seen.add(this)) return;
    output.add(NamedModule(prefix, this));
    for (final child in internalRegisteredChildren) {
      child.module._collectModules(_joinPath(prefix, child.name), output, seen);
    }
  }

  void _writeChildren(
    StringBuffer buffer,
    String prefix,
    List<NamedModule> entries,
  ) {
    for (var index = 0; index < entries.length; index++) {
      final entry = entries[index];
      final child = entry.module;
      child.internalEnsureMaterialized();
      final isLast = index == entries.length - 1;
      final connector = isLast ? '└── ' : '├── ';
      final labelPrefix = entry.name.isEmpty ? '' : '${entry.name}: ';
      buffer.writeln(
        '$prefix$connector$labelPrefix${child.internalDiagnosticLabel}',
      );
      final childPrefix = '$prefix${isLast ? '    ' : '│   '}';
      child._writeChildren(
        buffer,
        childPrefix,
        child.internalRegisteredChildren,
      );
    }
  }

  static String _joinPath(String prefix, String name) {
    if (name.isEmpty) return prefix;
    if (prefix.isEmpty) return name;
    return '$prefix.$name';
  }

  @override
  String toString() => internalDiagnosticLabel;
}
