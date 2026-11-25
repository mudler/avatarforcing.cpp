#pragma once
#include <cstdio>
#define AF_LOG(...) do { std::fprintf(stderr, "[avatarforcing] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
