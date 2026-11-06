include("${LOOM_SOURCE_DIR}/conformance/Conformance.cmake")
loom_add_conformance(second_provider "${LOOM_SOURCE_DIR}/conformance/mock_runtime.cpp")
# Also exercise declaration paths without inheriting the first sibling's state.
loom_declaration_tests()
