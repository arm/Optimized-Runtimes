cmake_minimum_required(VERSION 3.24)
project(or_interface LANGUAGES C CXX)
include(CTest)
include(conformance/Conformance.cmake)
if(BUILD_TESTING)
  or_declaration_tests()
  or_add_conformance(or_mock "${CMAKE_CURRENT_SOURCE_DIR}/conformance/mock_runtime.cpp")
  add_test(NAME or_mock COMMAND or_mock)
  add_test(NAME or_detect_wrong_output COMMAND or_mock --gtest_filter=HostConformance.RepeatedHostExecution)
  set_tests_properties(or_detect_wrong_output PROPERTIES ENVIRONMENT OR_MOCK_FAULT=wrong-output WILL_FAIL TRUE)
  add_test(NAME or_detect_clobbered_bootstrap COMMAND or_mock --gtest_filter=HostConformance.BootstrapVersionMismatchPreservesTable)
  set_tests_properties(or_detect_clobbered_bootstrap PROPERTIES ENVIRONMENT OR_MOCK_FAULT=clobber-bootstrap WILL_FAIL TRUE)
  # Independent input/output memory choices are part of the mandatory baseline.
  set(asymmetric_filter "HostConformance.CoreTableAndHostProfile:HostConformance.RepeatedHostExecution:HostConformance.IndependentSessionsExecuteConcurrently")
  foreach(profile IN ITEMS external-input external-output staged-input staged-output)
    add_test(NAME or_${profile} COMMAND or_mock "--gtest_filter=${asymmetric_filter}")
    set_tests_properties(or_${profile} PROPERTIES ENVIRONMENT "OR_MOCK_MEMORY_PROFILE=${profile}")
  endforeach()
  add_test(NAME or_detect_paired_memory COMMAND or_mock --gtest_filter=HostConformance.RepeatedHostExecution)
  set_tests_properties(or_detect_paired_memory PROPERTIES
    ENVIRONMENT "OR_MOCK_MEMORY_PROFILE=external-input;OR_MOCK_FAULT=paired-memory" WILL_FAIL TRUE)
  add_test(NAME or_single_session COMMAND or_mock)
  set_tests_properties(or_single_session PROPERTIES ENVIRONMENT "OR_MOCK_SESSION_PROFILE=single")
  foreach(method IN ITEMS borrow import)
    add_test(NAME or_larger_staged_${method} COMMAND or_mock
      --gtest_filter=HostConformance.AdvertisedStagedHostTransfer)
    set_tests_properties(or_larger_staged_${method} PROPERTIES
      ENVIRONMENT "OR_MOCK_LARGER_STAGED=1;OR_MOCK_HOST_METHOD=${method}")
  endforeach()
  add_test(NAME or_concurrent_stress COMMAND or_mock
    --gtest_filter=HostConformance.IndependentSessionsExecuteConcurrently --gtest_repeat=100)
  add_test(NAME or_sibling_consumer_configure COMMAND "${CMAKE_COMMAND}"
    -S "${CMAKE_CURRENT_SOURCE_DIR}/tests/sibling-consumer"
    -B "${CMAKE_CURRENT_BINARY_DIR}/sibling-consumer"
    -DOR_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR})
  get_property(or_tests DIRECTORY PROPERTY TESTS)
  set_tests_properties(${or_tests} PROPERTIES TIMEOUT 60)
  set_tests_properties(or_detect_wrong_output or_detect_clobbered_bootstrap
    or_detect_paired_memory PROPERTIES TIMEOUT 10)
endif()
