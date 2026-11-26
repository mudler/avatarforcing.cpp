add_executable(avatarforcing-cli main.cpp)
target_link_libraries(avatarforcing-cli PRIVATE avatarforcing)
target_include_directories(avatarforcing-cli SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/stb)
