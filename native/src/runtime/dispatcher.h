#ifndef TENSORA_RUNTIME_DISPATCHER_H_
#define TENSORA_RUNTIME_DISPATCHER_H_

#include "backends/backend.h"

namespace tensora {

class Dispatcher {
 public:
  static Status For(Device device, const Backend** out);
};

}  // namespace tensora

#endif  // TENSORA_RUNTIME_DISPATCHER_H_
