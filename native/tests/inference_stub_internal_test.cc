#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "inference/inference_bridge.h"

// Keep this internal contract self-contained instead of relying on C++ bridge
// symbols being exported from the Windows DLL. The public C ABI remains the
// DLL boundary; this test intentionally exercises the disabled backend bridge
// implementation directly on every platform.
#include "../src/core/status.cc"
#include "../src/inference/inference_bridge_stub.cc"

int main() {
  using namespace tensora::inference;
  std::string text;
  size_t count = 0;
  uint64_t session = 0;
  std::vector<std::shared_ptr<tensora::Tensor>> outputs;

  if (ProviderName(0, nullptr).code() != TS_INVALID_ARGUMENT) return 1;
  if (SessionCreate("", "", false, "", nullptr).code() != TS_INVALID_ARGUMENT)
    return 2;
  if (SessionProvider(0, nullptr).code() != TS_INVALID_ARGUMENT) return 3;
  if (SessionInputName(0, 0, nullptr).code() != TS_INVALID_ARGUMENT) return 4;
  if (SessionOutputName(0, 0, nullptr).code() != TS_INVALID_ARGUMENT) return 5;
  if (SessionRun(0, {}, {}, {}, nullptr).code() != TS_INVALID_ARGUMENT) return 6;
  if (SessionEndProfiling(0, nullptr).code() != TS_INVALID_ARGUMENT) return 7;

  if (ProviderCount(&count).code() != TS_OK || count != 0) return 8;
  if (ProviderName(0, &text).code() != TS_UNSUPPORTED || !text.empty()) return 9;
  if (SessionCreate("", "", false, "", &session).code() != TS_UNSUPPORTED ||
      session != 0)
    return 10;
  if (SessionProvider(0, &text).code() != TS_UNSUPPORTED || !text.empty())
    return 11;
  if (SessionInputCount(0, &count).code() != TS_UNSUPPORTED || count != 0)
    return 12;
  if (SessionOutputCount(0, &count).code() != TS_UNSUPPORTED || count != 0)
    return 13;
  if (SessionInputName(0, 0, &text).code() != TS_UNSUPPORTED || !text.empty())
    return 14;
  if (SessionOutputName(0, 0, &text).code() != TS_UNSUPPORTED || !text.empty())
    return 15;
  if (SessionRun(0, {}, {}, {}, &outputs).code() != TS_UNSUPPORTED ||
      !outputs.empty())
    return 16;
  if (SessionEndProfiling(0, &text).code() != TS_UNSUPPORTED || !text.empty())
    return 17;
  if (SessionRelease(0).code() != TS_UNSUPPORTED) return 18;
  return 0;
}
