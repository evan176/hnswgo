package hnswgo

/*
#cgo CXXFLAGS: -std=c++11
#include <stdlib.h>
#include <hnsw_wrapper.h>
*/
import "C"
import (
	"unsafe"
)

func New(dim int, m int, efConstruction int, randSeed int, maxElements uint32, spaceType string) {
	if spaceType == "ip" {
		C.initHNSW(C.int(dim), C.ulong(maxElements), C.int(m), C.int(efConstruction), C.int(randSeed), C.char('i'))
	} else if spaceType == "cosine" {
		C.initHNSW(C.int(dim), C.ulong(maxElements), C.int(m), C.int(efConstruction), C.int(randSeed), C.char('c'))
	} else {
		C.initHNSW(C.int(dim), C.ulong(maxElements), C.int(m), C.int(efConstruction), C.int(randSeed), C.char('l'))
	}
}

func Free() {
	C.freeHNSW()
}

func AddPoint(vector []float32, label uint32) {
	C.addPoint((*C.float)(unsafe.Pointer(&vector[0])), C.ulong(label))
}

func SearchKNN(vector []float32, k int) ([]uint32, []float32) {
	Clabel := make([]C.ulong, k, k)
	Cdist := make([]C.float, k, k)

	numResult := int(C.searchKNN((*C.float)(unsafe.Pointer(&vector[0])), C.int(k), &Clabel[0], &Cdist[0])) // perform the search

	labels := make([]uint32, k)
	dists := make([]float32, k)
	for i := 0; i < numResult; i++ {
		labels[i] = uint32(Clabel[i])
		dists[i] = float32(Cdist[i])
	}

	return labels[:numResult], dists[:numResult]
}

func SetEfConstruction(efConstruction int) {
	C.setEf(C.int(efConstruction))
}

//func Load(location string, dim int, spaceType string) *HNSW {
//	var hnsw HNSW
//	hnsw.dim = dim
//	hnsw.spaceType = spaceType
//
//	pLocation := C.CString(location)
//	if spaceType == "ip" {
//		hnsw.index = C.loadHNSW(pLocation, C.int(dim), C.char('i'))
//	} else if spaceType == "cosine" {
//		hnsw.normalize = true
//		hnsw.index = C.loadHNSW(pLocation, C.int(dim), C.char('i'))
//	} else {
//		hnsw.index = C.loadHNSW(pLocation, C.int(dim), C.char('l'))
//	}
//	C.free(unsafe.Pointer(pLocation))
//	return &hnsw
//}
//
//func (h *HNSW) Save(location string) {
//	pLocation := C.CString(location)
//	C.saveHNSW(h.index, pLocation)
//	C.free(unsafe.Pointer(pLocation))
//}

//func normalizeVector(vector []float32) []float32 {
//	var norm float32
//	for i := 0; i < len(vector); i++ {
//		norm += vector[i] * vector[i]
//	}
//	norm = 1.0 / (float32(math.Sqrt(float64(norm))) + 1e-15)
//	for i := 0; i < len(vector); i++ {
//		vector[i] = vector[i] * norm
//	}
//	return vector
//}
