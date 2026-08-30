#ifndef TENSORA_KERNELS_CPU_FEATURES_H_
#define TENSORA_KERNELS_CPU_FEATURES_H_

#include <cstdint>

namespace tensora::kernels {

/// Highest SIMD instruction set that is both compiled in and usable at runtime.
///
/// Tensora compiles one microkernel translation unit per level and selects the
/// implementation once per process. A binary built on a machine with AVX-512
/// must still run correctly on a machine without it, so the decision is never
/// made at compile time alone.
enum class SimdLevel : uint8_t {
  kScalar = 0,
  kAvx2 = 1,
  kAvx512 = 2,
};

/// Detects the usable SIMD level, including OS support for the register state.
///
/// The result is computed once and cached. Checking CPUID feature bits alone is
/// not sufficient: an operating system that does not save the wide register
/// state reports the instructions as present while executing them faults.
SimdLevel DetectSimdLevel();

/// Cached process-wide SIMD level.
SimdLevel CurrentSimdLevel();

/// Stable diagnostic name for a level.
const char* SimdLevelName(SimdLevel level);

/// Overrides the cached level. Testing hook: it allows the scalar and AVX2
/// paths to be exercised on a machine that supports AVX-512, so every compiled
/// microkernel is covered by the same numerical tests.
void SetSimdLevelForTesting(SimdLevel level);

}  // namespace tensora::kernels

#endif  // TENSORA_KERNELS_CPU_FEATURES_H_
