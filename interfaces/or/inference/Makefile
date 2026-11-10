# SPDX-License-Identifier: Apache-2.0
CXX ?= c++
OR_INCLUDE ?= ..
CPPFLAGS += -I$(OR_INCLUDE)
CXXFLAGS ?= -O2 -g
CXXFLAGS += -std=c++17 -Wall -Wextra -Wpedantic -Werror
LDLIBS += -ldl

.PHONY: all clean test
all: or-run

or-run: main.cpp $(OR_INCLUDE)/or_runtime.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp $(LDFLAGS) $(LDLIBS) -o $@

test_runtime.so: test_runtime.cpp $(OR_INCLUDE)/conformance/mock_runtime.cpp \
                 $(OR_INCLUDE)/conformance/provider.h $(OR_INCLUDE)/or_runtime.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -shared test_runtime.cpp \
	  $(OR_INCLUDE)/conformance/mock_runtime.cpp $(LDFLAGS) -o $@

test: or-run test_runtime.so
	python3 test_runner.py

clean:
	rm -f or-run test_runtime.so
