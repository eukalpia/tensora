/// Numerical data types understood by Tensora's public tensor semantics.
enum DType {
  /// IEEE-754 half-precision floating point.
  float16(byteWidth: 2, nativeCode: 2),

  /// Brain floating point with an eight-bit exponent.
  bfloat16(byteWidth: 2, nativeCode: 3),

  /// IEEE-754 single-precision floating point.
  float32(byteWidth: 4, nativeCode: 1),

  /// IEEE-754 double-precision floating point.
  float64(byteWidth: 8, nativeCode: 4),

  /// Signed eight-bit integer.
  int8(byteWidth: 1, nativeCode: 5),

  /// Unsigned eight-bit integer.
  uint8(byteWidth: 1, nativeCode: 6),

  /// Signed sixteen-bit integer.
  int16(byteWidth: 2, nativeCode: 7),

  /// Signed thirty-two-bit integer.
  int32(byteWidth: 4, nativeCode: 8),

  /// Signed sixty-four-bit integer.
  int64(byteWidth: 8, nativeCode: 9),

  /// Boolean tensor element.
  boolean(byteWidth: 1, nativeCode: 10);

  const DType({required this.byteWidth, required this.nativeCode});

  /// Storage width of one element in bytes.
  final int byteWidth;

  /// Stable value used by the C ABI.
  final int nativeCode;

  /// Whether this dtype is floating point.
  bool get isFloatingPoint => switch (this) {
    float16 || bfloat16 || float32 || float64 => true,
    _ => false,
  };

  /// Whether this dtype is an integer.
  bool get isInteger => switch (this) {
    int8 || uint8 || int16 || int32 || int64 => true,
    _ => false,
  };

  /// Whether this dtype stores boolean values.
  bool get isBoolean => this == boolean;

  /// Whether ordinary numerical arithmetic is defined for this dtype.
  bool get supportsArithmetic => !isBoolean;

  /// Whether gradient propagation is meaningful for this dtype.
  bool get supportsGradients => isFloatingPoint;

  /// Accumulator dtype used by reductions that can widen their inputs.
  DType get reductionAccumulator => switch (this) {
    float16 || bfloat16 || float32 => float32,
    float64 => float64,
    int8 || uint8 || int16 || int32 || int64 || boolean => int64,
  };

  /// Whether the current native tensor storage implements this dtype.
  ///
  /// The full semantic table is public now so operators, serialization, and
  /// future backends share one stable definition. Native allocation remains
  /// explicit and rejects values whose storage path is not implemented yet.
  bool get nativeStorageImplemented => this == float32;

  /// Returns the common dtype for a binary numerical operation.
  static DType promote(DType left, DType right) {
    if (left == right) return left;
    if (left.isBoolean) return right;
    if (right.isBoolean) return left;

    if (left.isFloatingPoint || right.isFloatingPoint) {
      if (left == float64 || right == float64) return float64;
      if (left == float32 || right == float32) return float32;
      if ((left == float16 && right == bfloat16) ||
          (left == bfloat16 && right == float16)) {
        return float32;
      }
      return left.isFloatingPoint ? left : right;
    }

    if ((left == uint8 && right == int8) || (left == int8 && right == uint8)) {
      return int16;
    }
    if (left == uint8) return right;
    if (right == uint8) return left;
    return left.byteWidth >= right.byteWidth ? left : right;
  }

  @override
  String toString() => 'DType.$name';
}
