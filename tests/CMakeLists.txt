function(af_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE avatarforcing)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  add_test(NAME ${name} COMMAND ${name})
  # Model-dependent tests exit 77 (skip) when their env vars are unset.
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

af_add_test(test_flow_parity)
set_tests_properties(test_flow_parity PROPERTIES LABELS "model" TIMEOUT 1800)

af_add_test(test_sampler_parity)
set_tests_properties(test_sampler_parity PROPERTIES LABELS "model" TIMEOUT 3600)

af_add_test(test_motion_parity)
set_tests_properties(test_motion_parity PROPERTIES LABELS "model" TIMEOUT 1800)

af_add_test(test_wav2vec2_parity)
set_tests_properties(test_wav2vec2_parity PROPERTIES LABELS "model" TIMEOUT 1800)

# Live mode feature schedule; needs no model.
af_add_test(test_live_schedule)

# The C API test is plain C and links the shared library.
if(TARGET avatarforcing_shared)
  add_executable(test_capi test_capi.c)
  target_link_libraries(test_capi PRIVATE avatarforcing_shared m)
  add_test(NAME test_capi COMMAND test_capi)
  set_tests_properties(test_capi PROPERTIES SKIP_RETURN_CODE 77 LABELS "model" TIMEOUT 1800)

  # mp4 output for speech shorter than the primer; needs ffmpeg too.
  add_executable(test_capi_mp4 test_capi_mp4.c)
  target_link_libraries(test_capi_mp4 PRIVATE avatarforcing_shared m)
  add_test(NAME test_capi_mp4 COMMAND test_capi_mp4)
  set_tests_properties(test_capi_mp4 PROPERTIES SKIP_RETURN_CODE 77 LABELS "model" TIMEOUT 1800)

  find_package(Threads REQUIRED)
  add_executable(test_capi_live test_capi_live.c)
  target_link_libraries(test_capi_live PRIVATE avatarforcing_shared Threads::Threads m)
  add_test(NAME test_capi_live COMMAND test_capi_live)
  set_tests_properties(test_capi_live PROPERTIES SKIP_RETURN_CODE 77 LABELS "model" TIMEOUT 3600)
endif()
