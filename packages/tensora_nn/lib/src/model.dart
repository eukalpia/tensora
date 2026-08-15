import 'package:tensora/tensora.dart' as core;

import 'module.dart';

/// Flutter-inspired declarative module with a lazily cached [build] tree.
abstract base class Model extends Module {
  Model();

  Module? _materializedRoot;
  bool _building = false;

  /// Builds the owned module tree for this model.
  Module build();

  @override
  bool get isMaterialized => _materializedRoot != null;

  @override
  List<NamedModule> get internalRegisteredChildren {
    final root = _materializedRoot;
    if (root == null) return const <NamedModule>[];
    return <NamedModule>[NamedModule('', root)];
  }

  @override
  void internalEnsureMaterialized() {
    if (_materializedRoot != null) return;
    if (_building) {
      throw core.InvalidArgumentException(
        'Model.build() recursively attempted to materialize the same model.',
        operation: 'model.build',
      );
    }

    _building = true;
    Module? candidate;
    var ownsCandidate = false;
    try {
      candidate = build();
      if (identical(candidate, this)) {
        throw core.InvalidArgumentException(
          'Model.build() cannot return the model itself.',
          operation: 'model.build',
        );
      }
      if (candidate.isDisposed) {
        throw core.InvalidArgumentException(
          'Model.build() returned a disposed module.',
          operation: 'model.build',
        );
      }
      if (candidate.internalOwner != null) {
        throw core.InvalidArgumentException(
          'Model.build() returned a module already owned by another path.',
          operation: 'model.build',
        );
      }
      if (candidate.internalContainsModule(this, <Module>{})) {
        throw core.InvalidArgumentException(
          'Model.build() created a cyclic module graph.',
          operation: 'model.build',
        );
      }

      candidate.internalAttachOwner(this);
      ownsCandidate = true;
      _materializedRoot = candidate;
    } catch (_) {
      if (candidate != null && ownsCandidate) {
        candidate.internalDetachOwner(this);
        candidate.dispose();
      }
      _materializedRoot = null;
      rethrow;
    } finally {
      _building = false;
    }
  }

  @override
  core.Tensor forward(core.Tensor input) {
    internalEnsureMaterialized();
    return _materializedRoot!(input);
  }

  @override
  String get internalDiagnosticLabel => runtimeType.toString();
}
