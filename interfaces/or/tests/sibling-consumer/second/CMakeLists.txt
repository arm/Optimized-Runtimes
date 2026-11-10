include("${OR_SOURCE_DIR}/conformance/Conformance.cmake")
or_add_conformance(second_provider "${OR_SOURCE_DIR}/conformance/mock_runtime.cpp")
# Also exercise declaration paths without inheriting the first sibling's state.
or_declaration_tests()
