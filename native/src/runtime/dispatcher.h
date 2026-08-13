#ifndef TENSORA_RUNTIME_DISPATCHER_H_
#define TENSORA_RUNTIME_DISPATCHER_H_

#include "backends/backend.h"

namespace tensora {

class Dispatcher {
 public:
  static Status For(Device device, const Backend** out);
  static Status ForTensor(const Tensor& tensor, const Backend** out);
  static Status ForTensors(const Tensor& left,
                           const Tensor& right,
                           const Backend** out);
};

}  // namespace tensora

#endif  // TENSORA_RUNTIME_DISPATCHER_H_
