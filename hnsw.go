package hnswgo

// #cgo LDFLAGS: -L${SRCDIR} -lhnsw -lm
// #include <stdlib.h>
// #include "hnsw_wrapper.h"
// HNSW initHNSW(int dim, unsigned long int max_elements, int M, int ef_construction, int rand_seed, char stype);
// HNSW loadHNSW(char *location, int dim, char stype);
// void addPoint(HNSW index, float *vec, unsigned long int label);
// int searchKnn(HNSW index, float *vec, int N, unsigned long int *result);
import "C"
import (
	"unsafe"
)

type HNSW struct {
	index     C.HNSW
	spaceType string
	dim       int
}

func New(dim, M, efConstruction, randSeed int, maxElements uint32, spaceType string) *HNSW {
	var hnsw HNSW
	hnsw.dim = dim
	hnsw.spaceType = spaceType
	if spaceType == "ip" {
		hnsw.index = C.initHNSW(C.int(dim), C.ulong(maxElements), C.int(M), C.int(efConstruction), C.int(randSeed), C.char('i'))
	} else {
		hnsw.index = C.initHNSW(C.int(dim), C.ulong(maxElements), C.int(M), C.int(efConstruction), C.int(randSeed), C.char('l'))
	}
	return &hnsw
}

func Load(location string, dim int, spaceType string) *HNSW {
	var hnsw HNSW
	hnsw.dim = dim
	hnsw.spaceType = spaceType

	pLocation := C.CString(location)
	if spaceType == "ip" {
		hnsw.index = C.loadHNSW(pLocation, C.int(dim), C.char('i'))
	} else {
		hnsw.index = C.loadHNSW(pLocation, C.int(dim), C.char('l'))
	}
	C.free(unsafe.Pointer(pLocation))
	return &hnsw
}

func (h *HNSW) Save(location string) {
	pLocation := C.CString(location)
	C.saveHNSW(h.index, pLocation)
	C.free(unsafe.Pointer(pLocation))
}

func (h *HNSW) AddPoint(vector []float32, label uint32) {
	C.addPoint(h.index, (*C.float)(unsafe.Pointer(&vector[0])), C.ulong(label))
}

func (h *HNSW) SearchKNN(vector []float32, N int) []uint32 {
	result := make([]C.ulong, N, N)
	numResult := int(C.searchKnn(h.index, (*C.float)(unsafe.Pointer(&vector[0])), C.int(N), &result[0]))
	labels := make([]uint32, N)
	for i := 0; i < numResult; i++ {
		labels[i] = uint32(result[i])
	}
	return labels[:numResult]
}
