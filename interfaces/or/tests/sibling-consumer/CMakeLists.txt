cmake_minimum_required(VERSION 3.24)
project(or_sibling_consumer LANGUAGES C CXX)
# Configuration-only regression: no dependency download or compilation needed.
# Each sibling must include the helper without relying on parent-scope state.
add_library(GTest::gtest_main INTERFACE IMPORTED GLOBAL)
add_subdirectory(first)
add_subdirectory(second)
