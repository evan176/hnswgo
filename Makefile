CXX=g++
INCLUDES=-I.
CXXFLAGS=-fPIC -pthread -Wall -std=c++0x -std=c++11 -O2 -march=native $(INCLUDES)
LDFLAGS=-shared
OBJS=hnsw_wrapper.o
TARGET=libhnsw.so

all: $(TARGET)

$(OBJS): hnsw_wrapper.h hnsw_wrapper.cc hnswlib/*.h
	$(CXX) $(CXXFLAGS) -c hnsw_wrapper.cc

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $(TARGET) $(OBJS)

clean:
	rm -rf $(OBJS) $(TARGET)
