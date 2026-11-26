#pragma once
#include <string>

namespace af {

// Prints "[avatarforcing] message" to stderr and keeps the message as the
// last log message of the calling thread. Error paths log the cause with
// AF_LOG and return false; the C API reads last_log() to report it.
#if defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif
void log(const char* fmt, ...);

// The last message logged on the calling thread ("" when none).
const std::string& last_log();
void clear_last_log();

} // namespace af

#define AF_LOG(...) ::af::log(__VA_ARGS__)
