#include "common.hpp"

#include <cstdarg>
#include <cstdio>

namespace af {

namespace {
thread_local std::string t_last_log;
}

void log(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    t_last_log = buf;
    std::fprintf(stderr, "[avatarforcing] %s\n", buf);
}

const std::string& last_log() { return t_last_log; }

void clear_last_log() { t_last_log.clear(); }

} // namespace af
