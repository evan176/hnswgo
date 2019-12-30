#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
CXX = c++
INCLUDES = -I.
CXXFLAGS = -pthread -std=c++0x -march=native -std=c++11 $(INCLUDES)
OBJS = hnsw_wrapper.o

opt: CXXFLAGS += -O3 -funroll-loops
opt: build

coverage: CXXFLAGS += -O0 -fno-inline -fprofile-arcs --coverage

hnsw_wrapper.o: hnsw_wrapper.h hnsw_wrapper.cc hnswlib/hnswlib/*.h
	$(CXX) $(CXXFLAGS) -c hnsw_wrapper.cc

libhnsw.a: $(OBJS)
	$(AR) rcs libhnsw.a $(OBJS)

clean:
	rm -rf *.o libhnsw.a *.o *.gcno *.gcda hnsw

build: libhnsw.a
	env CGO_CXXFLAGS="$(INCLUDES) -std=c++11" go build

test: build
	go test
