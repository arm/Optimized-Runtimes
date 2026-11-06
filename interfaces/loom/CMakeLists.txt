cmake_minimum_required(VERSION 3.24)
project(loom_interface LANGUAGES C CXX)
include(CTest)
include(conformance/Conformance.cmake)
if(BUILD_TESTING)
  loom_declaration_tests()
  loom_add_conformance(loom_mock "${CMAKE_CURRENT_SOURCE_DIR}/conformance/mock_runtime.cpp")
  add_test(NAME loom_mock COMMAND loom_mock)
  add_test(NAME loom_detect_wrong_output COMMAND loom_mock --gtest_filter=HostConformance.RepeatedHostExecution)
  set_tests_properties(loom_detect_wrong_output PROPERTIES ENVIRONMENT LRT_MOCK_FAULT=wrong-output WILL_FAIL TRUE)
  add_test(NAME loom_detect_clobbered_bootstrap COMMAND loom_mock --gtest_filter=HostConformance.BootstrapVersionMismatchPreservesTable)
  set_tests_properties(loom_detect_clobbered_bootstrap PROPERTIES ENVIRONMENT LRT_MOCK_FAULT=clobber-bootstrap WILL_FAIL TRUE)
  # Independent input/output memory choices are part of the mandatory baseline.
  set(asymmetric_filter "HostConformance.CoreTableAndHostProfile:HostConformance.RepeatedHostExecution:HostConformance.IndependentSessionsExecuteConcurrently")
  foreach(profile IN ITEMS external-input external-output staged-input staged-output)
    add_test(NAME loom_${profile} COMMAND loom_mock "--gtest_filter=${asymmetric_filter}")
    set_tests_properties(loom_${profile} PROPERTIES ENVIRONMENT "LRT_MOCK_MEMORY_PROFILE=${profile}")
  endforeach()
  add_test(NAME loom_detect_paired_memory COMMAND loom_mock --gtest_filter=HostConformance.RepeatedHostExecution)
  set_tests_properties(loom_detect_paired_memory PROPERTIES
    ENVIRONMENT "LRT_MOCK_MEMORY_PROFILE=external-input;LRT_MOCK_FAULT=paired-memory" WILL_FAIL TRUE)
  add_test(NAME loom_single_session COMMAND loom_mock)
  set_tests_properties(loom_single_session PROPERTIES ENVIRONMENT "LRT_MOCK_SESSION_PROFILE=single")
  foreach(method IN ITEMS borrow import)
    add_test(NAME loom_larger_staged_${method} COMMAND loom_mock
      --gtest_filter=HostConformance.AdvertisedStagedHostTransfer)
    set_tests_properties(loom_larger_staged_${method} PROPERTIES
      ENVIRONMENT "LRT_MOCK_LARGER_STAGED=1;LRT_MOCK_HOST_METHOD=${method}")
  endforeach()
  add_test(NAME loom_concurrent_stress COMMAND loom_mock
    --gtest_filter=HostConformance.IndependentSessionsExecuteConcurrently --gtest_repeat=100)
  add_test(NAME loom_sibling_consumer_configure COMMAND "${CMAKE_COMMAND}"
    -S "${CMAKE_CURRENT_SOURCE_DIR}/tests/sibling-consumer"
    -B "${CMAKE_CURRENT_BINARY_DIR}/sibling-consumer"
    -DLOOM_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR})
  get_property(loom_tests DIRECTORY PROPERTY TESTS)
  set_tests_properties(${loom_tests} PROPERTIES TIMEOUT 60)
  set_tests_properties(loom_detect_wrong_output loom_detect_clobbered_bootstrap
    loom_detect_paired_memory PROPERTIES TIMEOUT 10)
endif()
