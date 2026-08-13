#ifndef TENSORA_RUNTIME_DEVICE_CODEC_H_
#define TENSORA_RUNTIME_DEVICE_CODEC_H_

#include <cstdint>

#include "core/status.h"
#include "tensor/tensor.h"

namespace tensora {

Status DeviceFromCode(uint32_t code, Device* out_device);

}  // namespace tensora

#endif  // TENSORA_RUNTIME_DEVICE_CODEC_H_
