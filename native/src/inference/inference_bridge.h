#ifndef TENSORA_INFERENCE_INFERENCE_BRIDGE_H_
#define TENSORA_INFERENCE_INFERENCE_BRIDGE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/status.h"
#include "tensor/tensor.h"

namespace tensora::inference {

Status IsAvailable(uint8_t* out_available);
Status ProviderCount(size_t* out_count);
Status ProviderName(size_t index, std::string* out_name);

Status SessionCreate(const std::string& model_path,
                     const std::string& requested_provider,
                     bool enable_profiling,
                     const std::string& profiling_prefix,
                     uint64_t* out_session);
Status SessionProvider(uint64_t session, std::string* out_provider);
Status SessionInputCount(uint64_t session, size_t* out_count);
Status SessionOutputCount(uint64_t session, size_t* out_count);
Status SessionInputName(uint64_t session, size_t index, std::string* out_name);
Status SessionOutputName(uint64_t session, size_t index, std::string* out_name);
Status SessionRun(uint64_t session,
                  const std::vector<std::string>& input_names,
                  const std::vector<std::shared_ptr<Tensor>>& inputs,
                  const std::vector<std::string>& output_names,
                  std::vector<std::shared_ptr<Tensor>>* out_tensors);
Status SessionEndProfiling(uint64_t session, std::string* out_path);
Status SessionRelease(uint64_t session);

}  // namespace tensora::inference

#endif  // TENSORA_INFERENCE_INFERENCE_BRIDGE_H_
