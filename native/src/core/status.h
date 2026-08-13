#ifndef TENSORA_CORE_STATUS_H_
#define TENSORA_CORE_STATUS_H_

#include <string>

#include "tensora.h"

namespace tensora {

class Status {
 public:
  Status() = default;
  Status(ts_status_t code, std::string message)
      : code_(code), message_(std::move(message)) {}

  static Status Ok() { return Status(); }

  bool ok() const { return code_ == TS_OK; }
  ts_status_t code() const { return code_; }
  const std::string& message() const { return message_; }

 private:
  ts_status_t code_ = TS_OK;
  std::string message_;
};

Status InvalidArgument(std::string message);
Status InvalidShape(std::string message);
Status OutOfMemory(std::string message);
Status Unsupported(std::string message);
Status InvalidHandle(std::string message);
Status InternalError(std::string message);
Status ModelError(std::string message);

void ClearLastError();
void SetLastError(const Status& status);
const char* LastErrorMessage();

}  // namespace tensora

#endif  // TENSORA_CORE_STATUS_H_
