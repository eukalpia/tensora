/// Base type for recoverable Tensora failures.
sealed class TensoraException implements Exception {
  const TensoraException(
    this.message, {
    this.operation,
  });

  /// Human-readable diagnostic message.
  final String message;

  /// Tensora operation associated with the failure, when known.
  final String? operation;

  @override
  String toString() {
    final prefix = operation == null ? runtimeType : '$runtimeType($operation)';
    return '$prefix: $message';
  }
}

/// A public API or native ABI argument was invalid.
final class InvalidArgumentException extends TensoraException {
  const InvalidArgumentException(super.message, {super.operation});
}

/// Tensor dimensions or shape compatibility were invalid.
final class InvalidShapeException extends TensoraException {
  const InvalidShapeException(super.message, {super.operation});
}

/// Tensora could not allocate required native memory.
final class OutOfMemoryException extends TensoraException {
  const OutOfMemoryException(super.message, {super.operation});
}

/// The requested operation or runtime capability is unsupported.
final class UnsupportedOperationException extends TensoraException {
  const UnsupportedOperationException(super.message, {super.operation});
}

/// The loaded native runtime failed or violated the expected ABI contract.
final class NativeRuntimeException extends TensoraException {
  const NativeRuntimeException(super.message, {super.operation});
}

/// A Tensor was used after deterministic disposal.
final class DisposedTensorException extends TensoraException {
  const DisposedTensorException(super.message, {super.operation});
}
