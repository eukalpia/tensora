#include "kernels/cpu_features.h"

#include <atomic>

#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace tensora::kernels {
namespace {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define TENSORA_X86 1
#endif

#if defined(TENSORA_X86)

struct CpuidRegisters {
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;
};

CpuidRegisters Cpuid(uint32_t leaf, uint32_t subleaf) {
  CpuidRegisters registers;
#if defined(_MSC_VER)
  int values[4] = {0, 0, 0, 0};
  __cpuidex(values, static_cast<int>(leaf), static_cast<int>(subleaf));
  registers.eax = static_cast<uint32_t>(values[0]);
  registers.ebx = static_cast<uint32_t>(values[1]);
  registers.ecx = static_cast<uint32_t>(values[2]);
  registers.edx = static_cast<uint32_t>(values[3]);
#else
  __cpuid_count(leaf, subleaf, registers.eax, registers.ebx, registers.ecx,
                registers.edx);
#endif
  return registers;
}

uint64_t ExtendedControlRegister0() {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  uint32_t low = 0;
  uint32_t high = 0;
  __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0"
                       : "=a"(low), "=d"(high)
                       : "c"(0));
  return (static_cast<uint64_t>(high) << 32) | low;
#endif
}

bool HasBit(uint32_t value, uint32_t bit) {
  return (value & (UINT32_C(1) << bit)) != 0;
}

SimdLevel Detect() {
  const CpuidRegisters leaf0 = Cpuid(0, 0);
  if (leaf0.eax < 1) return SimdLevel::kScalar;

  const CpuidRegisters leaf1 = Cpuid(1, 0);
  const bool has_osxsave = HasBit(leaf1.ecx, 27);
  const bool has_avx = HasBit(leaf1.ecx, 28);
  const bool has_fma = HasBit(leaf1.ecx, 12);
  if (!has_osxsave || !has_avx || !has_fma) return SimdLevel::kScalar;

  // The OS must actually preserve the wide register state across context
  // switches. XMM is bit 1, YMM is bit 2, and the AVX-512 opmask/ZMM state
  // occupies bits 5 through 7.
  const uint64_t xcr0 = ExtendedControlRegister0();
  const bool ymm_saved = (xcr0 & 0x6) == 0x6;
  if (!ymm_saved) return SimdLevel::kScalar;

  if (leaf0.eax < 7) return SimdLevel::kAvx2;
  const CpuidRegisters leaf7 = Cpuid(7, 0);
  if (!HasBit(leaf7.ebx, 5)) return SimdLevel::kScalar;  // AVX2

  const bool zmm_saved = (xcr0 & 0xe6) == 0xe6;
  const bool has_avx512f = HasBit(leaf7.ebx, 16);
  const bool has_avx512dq = HasBit(leaf7.ebx, 17);
  const bool has_avx512bw = HasBit(leaf7.ebx, 30);
  const bool has_avx512vl = HasBit(leaf7.ebx, 31);
  if (zmm_saved && has_avx512f && has_avx512dq && has_avx512bw &&
      has_avx512vl) {
    return SimdLevel::kAvx512;
  }
  return SimdLevel::kAvx2;
}

#else  // !TENSORA_X86

SimdLevel Detect() { return SimdLevel::kScalar; }

#endif  // TENSORA_X86

std::atomic<uint8_t>& LevelStorage() {
  static std::atomic<uint8_t> level{
      static_cast<uint8_t>(0xffu)};
  return level;
}

}  // namespace

SimdLevel DetectSimdLevel() { return Detect(); }

SimdLevel CurrentSimdLevel() {
  auto& storage = LevelStorage();
  const uint8_t cached = storage.load(std::memory_order_acquire);
  if (cached != 0xffu) return static_cast<SimdLevel>(cached);
  const SimdLevel detected = Detect();
  storage.store(static_cast<uint8_t>(detected), std::memory_order_release);
  return detected;
}

void SetSimdLevelForTesting(SimdLevel level) {
  LevelStorage().store(static_cast<uint8_t>(level), std::memory_order_release);
}

const char* SimdLevelName(SimdLevel level) {
  switch (level) {
    case SimdLevel::kScalar:
      return "scalar";
    case SimdLevel::kAvx2:
      return "avx2";
    case SimdLevel::kAvx512:
      return "avx512";
  }
  return "unknown";
}

}  // namespace tensora::kernels
