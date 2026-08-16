import 'dart:typed_data';

int _float32Bits(double value) {
  final bytes = ByteData(4)..setFloat32(0, value, Endian.host);
  return bytes.getUint32(0, Endian.host);
}

double _float32FromBits(int bits) {
  final bytes = ByteData(4)..setUint32(0, bits & 0xffffffff, Endian.host);
  return bytes.getFloat32(0, Endian.host);
}

/// Encodes [value] as an IEEE-754 binary16 bit pattern using
/// round-to-nearest-even.
int encodeFloat16(double value) {
  final bits = _float32Bits(value);
  final sign = (bits >> 16) & 0x8000;
  final exponent = (bits >> 23) & 0xff;
  var mantissa = bits & 0x007fffff;

  if (exponent == 0xff) {
    if (mantissa == 0) return sign | 0x7c00;
    var payload = mantissa >> 13;
    if (payload == 0) payload = 1;
    return sign | 0x7c00 | payload | 0x0200;
  }

  var targetExponent = exponent - 127 + 15;
  if (targetExponent >= 31) return sign | 0x7c00;
  if (targetExponent <= 0) {
    if (targetExponent < -10) return sign;
    mantissa |= 0x00800000;
    final shift = 14 - targetExponent;
    var rounded = mantissa >> shift;
    final remainderMask = (1 << shift) - 1;
    final remainder = mantissa & remainderMask;
    final halfway = 1 << (shift - 1);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & 1) != 0)) {
      rounded++;
    }
    return sign | rounded;
  }

  var roundedMantissa = mantissa >> 13;
  final remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 ||
      (remainder == 0x1000 && (roundedMantissa & 1) != 0)) {
    roundedMantissa++;
    if (roundedMantissa == 0x0400) {
      roundedMantissa = 0;
      targetExponent++;
      if (targetExponent >= 31) return sign | 0x7c00;
    }
  }
  return sign | (targetExponent << 10) | roundedMantissa;
}

/// Decodes an IEEE-754 binary16 bit pattern into a Dart double containing the
/// exactly representable float32 value.
double decodeFloat16(int value) {
  final normalized = value & 0xffff;
  final sign = (normalized & 0x8000) << 16;
  final exponent = (normalized >> 10) & 0x1f;
  var mantissa = normalized & 0x03ff;
  late int bits;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      var adjustedExponent = -14;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        adjustedExponent--;
      }
      mantissa &= 0x03ff;
      bits =
          sign |
          ((adjustedExponent + 127) << 23) |
          (mantissa << 13);
    }
  } else if (exponent == 0x1f) {
    bits = sign | 0x7f800000 | (mantissa << 13);
    if (mantissa != 0) bits |= 0x00400000;
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  return _float32FromBits(bits);
}

/// Encodes [value] as a bfloat16 bit pattern using round-to-nearest-even.
int encodeBFloat16(double value) {
  var bits = _float32Bits(value);
  if ((bits & 0x7f800000) == 0x7f800000 &&
      (bits & 0x007fffff) != 0) {
    return ((bits >> 16) | 0x0040) & 0xffff;
  }
  final leastSignificant = (bits >> 16) & 1;
  bits = (bits + 0x00007fff + leastSignificant) & 0xffffffff;
  return (bits >> 16) & 0xffff;
}

/// Decodes a bfloat16 bit pattern into a Dart double containing the exactly
/// representable float32 value.
double decodeBFloat16(int value) =>
    _float32FromBits((value & 0xffff) << 16);
