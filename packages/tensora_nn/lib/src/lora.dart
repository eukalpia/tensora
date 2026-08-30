import 'dart:math' as math;

import 'package:tensora/tensora.dart' as core;

import 'layers.dart';
import 'module.dart';

/// Low-rank adaptation of a frozen linear projection.
///
/// The base projection keeps its weights and its native parameter identity, and
/// is frozen. Only the two low-rank factors train, so an optimizer built from
/// [parameters] touches `rank * (inFeatures + outFeatures)` values instead of
/// `inFeatures * outFeatures`. That ratio is the whole point of the technique:
/// it is what makes fine-tuning a large projection affordable.
///
/// The forward pass computes `base(x) + scaling * ((x @ A) @ B)` where
/// `scaling` is `alpha / rank`. Following the original formulation, `B` starts
/// at zero, so a freshly constructed adapter is exactly the identity and
/// training begins from the base model's behavior rather than from noise.
///
/// The factors are ordinary [Linear] modules without bias. That is deliberate:
/// they inherit native parameter identity and the transactional device move
/// that a hand-rolled tensor pair would not have.
final class LoRALinear extends Module {
  /// Creates an adapter around a new frozen [Linear] projection.
  ///
  /// [rank] must be positive and no larger than either feature dimension;
  /// a rank above that stores more values than the projection it adapts.
  factory LoRALinear({
    required int inFeatures,
    required int outFeatures,
    required int rank,
    double alpha = 1,
    bool bias = true,
    int? seed,
  }) {
    if (rank <= 0) {
      throw ArgumentError.value(rank, 'rank', 'must be positive');
    }
    if (rank > inFeatures || rank > outFeatures) {
      throw ArgumentError.value(
        rank,
        'rank',
        'must not exceed either feature dimension '
            '($inFeatures, $outFeatures)',
      );
    }
    if (!alpha.isFinite || alpha <= 0) {
      throw ArgumentError.value(alpha, 'alpha', 'must be finite and positive');
    }

    final base = Linear(
      inFeatures: inFeatures,
      outFeatures: outFeatures,
      bias: bias,
    );
    Linear? down;
    Linear? up;
    try {
      down = Linear(inFeatures: inFeatures, outFeatures: rank, bias: false);
      up = Linear(inFeatures: rank, outFeatures: outFeatures, bias: false);

      // The base projection is what we are adapting, not what we are training.
      for (final entry in base.namedParameters) {
        entry.parameter.freeze();
      }

      _initializeFactors(down, up, inFeatures: inFeatures, seed: seed);

      final adapter = LoRALinear._(
        base,
        down,
        up,
        inFeatures: inFeatures,
        outFeatures: outFeatures,
        rank: rank,
        alpha: alpha,
        bias: bias,
      );
      base.internalAttachOwner(adapter);
      down.internalAttachOwner(adapter);
      up.internalAttachOwner(adapter);
      return adapter;
    } catch (_) {
      up?.dispose();
      down?.dispose();
      base.dispose();
      rethrow;
    }
  }

  LoRALinear._(
    this._base,
    this._down,
    this._up, {
    required this.inFeatures,
    required this.outFeatures,
    required this.rank,
    required this.alpha,
    required this.bias,
  }) : _entries = List<NamedModule>.unmodifiable(<NamedModule>[
         NamedModule('base', _base),
         NamedModule('loraA', _down),
         NamedModule('loraB', _up),
       ]);

  final Linear _base;
  final Linear _down;
  final Linear _up;
  final List<NamedModule> _entries;

  /// Input width of the adapted projection.
  final int inFeatures;

  /// Output width of the adapted projection.
  final int outFeatures;

  /// Inner dimension of the low-rank update.
  final int rank;

  /// Numerator of the update scaling.
  final double alpha;

  /// Whether the frozen base projection owns a bias.
  final bool bias;

  core.Tensor? _scaleCache;

  /// Constant applied to the low-rank update, `alpha / rank`.
  ///
  /// Keeping it separate from the factors is what lets [rank] change without
  /// retuning the learning rate.
  double get scaling => alpha / rank;

  /// The frozen projection being adapted.
  Linear get base => _base;

  @override
  List<NamedModule> get internalRegisteredChildren => _entries;

  @override
  core.Tensor forward(core.Tensor input) {
    final projected = _base(input);
    core.Tensor? low;
    core.Tensor? update;
    core.Tensor? scaled;
    try {
      low = _down(input);
      scaled = low.multiply(_scaleFor(low));
      update = _up(scaled);
      return projected.add(update);
    } finally {
      update?.dispose();
      scaled?.dispose();
      low?.dispose();
      projected.dispose();
    }
  }

  /// Folds the adapter into the base weights and zeroes the factors.
  ///
  /// After merging, [forward] returns the same values while the low-rank path
  /// contributes nothing, so inference costs exactly what the unadapted
  /// projection costs. Parameter identities are preserved throughout, which
  /// means an optimizer built before the merge stays valid.
  void mergeIntoBase() {
    _ensureUsable('mergeIntoBase');

    final baseWeight =
        _base.namedParameters
            .firstWhere((entry) => entry.name == 'weight')
            .parameter
            .tensorForRuntime;
    final downWeight = _down.namedParameters.single.parameter.tensorForRuntime;
    final upWeight = _up.namedParameters.single.parameter.tensorForRuntime;

    // The two native backends disagree on weight orientation: the core build
    // stores [in, out] while LibTorch follows the PyTorch convention of
    // [out, in]. Rather than assume one, read the orientation off the factors
    // and compose in whichever order reproduces the base weight's shape.
    final rowsAreInput = _storesInputMajor(downWeight, baseWeight);

    core.Tensor? product;
    core.Tensor? scale;
    core.Tensor? scaledProduct;
    core.Tensor? merged;
    try {
      product =
          rowsAreInput
              ? downWeight.matmul(upWeight)
              : upWeight.matmul(downWeight);
      if (product.shape != baseWeight.shape) {
        throw core.NativeRuntimeException(
          'LoRA factors compose to ${product.shape}, which does not match the '
          'base weight ${baseWeight.shape}.',
          operation: 'lora.mergeIntoBase',
        );
      }
      scale = core.Tensor.full(product.shape, scaling, device: product.device);
      scaledProduct = product.multiply(scale);
      merged = baseWeight.add(scaledProduct);
      core.NativeTensorState.assignMany(
        targets: <core.Tensor>[baseWeight],
        sources: <core.Tensor>[merged],
      );
    } finally {
      merged?.dispose();
      scaledProduct?.dispose();
      scale?.dispose();
      product?.dispose();
    }

    // Zeroing the up factor makes the merge idempotent: calling it twice does
    // not apply the update twice.
    final zeros = _matchBackend(
      upWeight,
      core.Tensor.zeros(upWeight.shape, device: upWeight.device),
    );
    try {
      core.NativeTensorState.assignMany(
        targets: <core.Tensor>[upWeight],
        sources: <core.Tensor>[zeros],
      );
    } finally {
      zeros.dispose();
    }
  }

  @override
  void internalOnDispose() {
    _scaleCache?.dispose();
    _scaleCache = null;
  }

  @override
  String get internalDiagnosticLabel =>
      'LoRALinear(inFeatures: $inFeatures, outFeatures: $outFeatures, '
      'rank: $rank, alpha: $alpha, bias: $bias)';

  /// Returns a cached constant tensor matching [reference], rebuilt only when
  /// the batch shape or device changes.
  core.Tensor _scaleFor(core.Tensor reference) {
    final cached = _scaleCache;
    if (cached != null &&
        !cached.isDisposed &&
        cached.shape == reference.shape &&
        cached.device == reference.device) {
      return cached;
    }
    cached?.dispose();
    final created = core.Tensor.full(
      reference.shape,
      scaling,
      device: reference.device,
    );
    _scaleCache = created;
    return created;
  }

  /// Whether native module weights place the input dimension first.
  ///
  /// The dependency-light build stores a `Linear` weight as
  /// `[inFeatures, outFeatures]`; a LibTorch build follows the PyTorch
  /// convention and stores `[outFeatures, inFeatures]`. The two are told apart
  /// by the down factor, whose shape involves `rank`, and by the base weight
  /// when the factor alone is ambiguous.
  bool _storesInputMajor(core.Tensor downWeight, core.Tensor baseWeight) {
    if (rank != inFeatures) {
      return downWeight.shape.dimensions[0] == inFeatures;
    }
    if (inFeatures != outFeatures) {
      return baseWeight.shape.dimensions[0] == inFeatures;
    }
    throw core.UnsupportedOperationException(
      'Cannot determine the native weight orientation when rank, inFeatures '
      'and outFeatures are all $rank.',
      operation: 'lora.mergeIntoBase',
    );
  }

  /// Transfers [created] onto the device that owns [reference], taking
  /// ownership of it.
  ///
  /// This is not redundant when the devices already match. A LibTorch build
  /// keeps module parameters in Torch storage while host factories produce core
  /// CPU storage, and a state assignment requires both sides to sit on the same
  /// backend. The transfer is what normalizes them.
  static core.Tensor _matchBackend(core.Tensor reference, core.Tensor created) {
    try {
      return created.to(reference.device);
    } finally {
      created.dispose();
    }
  }

  void _ensureUsable(String operation) {
    if (isDisposed) {
      throw core.DisposedTensorException(
        'LoRALinear has already been disposed.',
        operation: 'lora.$operation',
      );
    }
  }

  /// Applies the reference initialization: the down factor is drawn from a
  /// bounded uniform distribution, the up factor is exactly zero.
  static void _initializeFactors(
    Linear down,
    Linear up, {
    required int inFeatures,
    required int? seed,
  }) {
    final downWeight = down.namedParameters.single.parameter.tensorForRuntime;
    final upWeight = up.namedParameters.single.parameter.tensorForRuntime;

    final random = math.Random(seed ?? 0x5eed);
    final bound = 1.0 / math.sqrt(inFeatures);
    final values = List<double>.generate(
      downWeight.shape.numel,
      (_) => (random.nextDouble() * 2 - 1) * bound,
      growable: false,
    );

    core.Tensor? sampled;
    core.Tensor? zeros;
    try {
      sampled = _matchBackend(
        downWeight,
        core.Tensor.fromList(
          values,
          shape: downWeight.shape,
          device: downWeight.device,
        ),
      );
      zeros = _matchBackend(
        upWeight,
        core.Tensor.zeros(upWeight.shape, device: upWeight.device),
      );
      core.NativeTensorState.assignMany(
        targets: <core.Tensor>[downWeight, upWeight],
        sources: <core.Tensor>[sampled, zeros],
      );
    } finally {
      zeros?.dispose();
      sampled?.dispose();
    }
  }
}
