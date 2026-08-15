import 'package:tensora/tensora.dart' as core;

import 'module.dart';
import 'parameter.dart';

/// Executes child modules in order.
final class Sequential extends Module {
  factory Sequential({required List<Module> children}) {
    final frozen = List<Module>.unmodifiable(children);
    final sequential = Sequential._(frozen);
    final attached = <Module>[];
    try {
      for (final child in frozen) {
        if (child.isDisposed) {
          throw ArgumentError('Sequential cannot own a disposed child.');
        }
        child.internalAttachOwner(sequential);
        attached.add(child);
      }
      return sequential;
    } catch (_) {
      for (final child in attached.reversed) {
        child.internalDetachOwner(sequential);
      }
      rethrow;
    }
  }

  Sequential._(List<Module> children)
    : _children = children,
      _entries = List<NamedModule>.unmodifiable(<NamedModule>[
        for (var index = 0; index < children.length; index++)
          NamedModule('$index', children[index]),
      ]);

  final List<Module> _children;
  final List<NamedModule> _entries;

  @override
  List<Module> get children {
    if (isDisposed) {
      throw core.DisposedTensorException(
        'Module has already been disposed.',
        operation: 'module.children',
      );
    }
    return _children;
  }

  @override
  List<NamedModule> get internalRegisteredChildren => _entries;

  @override
  core.Tensor forward(core.Tensor input) {
    var current = input;
    for (final child in _children) {
      core.Tensor next;
      try {
        next = child(current);
      } catch (_) {
        if (!identical(current, input)) current.dispose();
        rethrow;
      }
      if (!identical(current, input) && !identical(current, next)) {
        current.dispose();
      }
      current = next;
    }
    return current;
  }

  @override
  String get internalDiagnosticLabel => 'Sequential';
}

/// Identity module. Returns the same Tensor wrapper unchanged.
final class Identity extends Module {
  Identity();

  @override
  core.Tensor forward(core.Tensor input) => input;

  @override
  String get internalDiagnosticLabel => 'Identity()';
}

/// Native-backed fully connected affine layer.
final class Linear extends Module {
  factory Linear({
    required int inFeatures,
    required int outFeatures,
    bool bias = true,
  }) {
    if (inFeatures <= 0) {
      throw ArgumentError.value(inFeatures, 'inFeatures', 'must be positive');
    }
    if (outFeatures <= 0) {
      throw ArgumentError.value(outFeatures, 'outFeatures', 'must be positive');
    }

    final native = core.Linear(inFeatures, outFeatures, bias: bias);
    final tensors = <core.Tensor>[];
    try {
      tensors.addAll(native.parameters());
      final expected = bias ? 2 : 1;
      if (tensors.length != expected) {
        throw core.NativeRuntimeException(
          'Linear exposed ${tensors.length} parameters; expected $expected.',
          operation: 'nn.linear.create',
        );
      }
      final parameters = <NamedParameter>[
        NamedParameter('weight', Parameter.fromTensor(tensors[0])),
        if (bias) NamedParameter('bias', Parameter.fromTensor(tensors[1])),
      ];
      return Linear._(
        native,
        inFeatures: inFeatures,
        outFeatures: outFeatures,
        bias: bias,
        parameters: List<NamedParameter>.unmodifiable(parameters),
      );
    } catch (_) {
      for (final tensor in tensors) {
        tensor.dispose();
      }
      native.dispose();
      rethrow;
    }
  }

  Linear._(
    this._native, {
    required this.inFeatures,
    required this.outFeatures,
    required this.bias,
    required List<NamedParameter> parameters,
  }) : _parameters = parameters;

  final core.Linear _native;
  final List<NamedParameter> _parameters;

  final int inFeatures;
  final int outFeatures;
  final bool bias;

  @override
  List<NamedParameter> get internalRegisteredParameters => _parameters;

  @override
  core.Tensor forward(core.Tensor input) => _native(input);

  @override
  void internalOnTrainingModeChanged(bool training) {
    if (training) {
      _native.train();
    } else {
      _native.eval();
    }
  }

  @override
  void internalOnMove(core.Device device) => _native.to(device);

  @override
  void internalOnDispose() => _native.dispose();

  @override
  String get internalDiagnosticLabel =>
      'Linear(inFeatures: $inFeatures, outFeatures: $outFeatures, bias: $bias)';
}
