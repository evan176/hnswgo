package hnswgo

/*
#cgo CXXFLAGS: -std=c++11
#include <stdlib.h>
#include <hnsw_wrapper.h>
*/
import "C"
import (
	"math"
	"unsafe"
)

type Index struct {
	index      C.HNSW
	dimensions int
	normalize  bool
	spaceType  string
}

func normalize(vector []float32) { // normalize(v) = (1/|v|)*v
	var magnitude float32
	for i := range vector {
		magnitude += vector[i] * vector[i]
	}
	magnitude = float32(math.Sqrt(float64(magnitude)))

	for i := range vector {
		vector[i] *= 1 / magnitude
	}
}

func New(dim int, m int, efConstruction int, randSeed int, maxElements uint32, spaceType string) *Index {
	index := new(Index)
	index.dimensions = dim
	index.spaceType = spaceType
	if spaceType == "ip" {
		index.index = C.initHNSW(C.int(dim), C.ulong(maxElements), C.int(m), C.int(efConstruction), C.int(randSeed), C.char('i'))
	} else if spaceType == "cosine" {
		index.normalize = true
		index.index = C.initHNSW(C.int(dim), C.ulong(maxElements), C.int(m), C.int(efConstruction), C.int(randSeed), C.char('c'))
	} else {
		index.index = C.initHNSW(C.int(dim), C.ulong(maxElements), C.int(m), C.int(efConstruction), C.int(randSeed), C.char('l'))
	}
	return index
}

func (i *Index) Free() {
	C.freeHNSW(i.index)
}

func (i *Index) AddPoint(vector []float32, label uint32) {
	if i.normalize {
		normalize(vector)
	}
	C.addPoint(i.index, (*C.float)(unsafe.Pointer(&vector[0])), C.ulong(label))
}

func (i *Index) SearchKNN(vector []float32, k int) ([]uint32, []float32) {
	if i.normalize {
		normalize(vector)
	}

	Clabel := make([]C.ulong, k, k)
	Cdist := make([]C.float, k, k)

	numResult := int(C.searchKNN(i.index, (*C.float)(unsafe.Pointer(&vector[0])), C.int(k), &Clabel[0], &Cdist[0])) // perform the search

	labels := make([]uint32, k)
	dists := make([]float32, k)
	for i := 0; i < numResult; i++ {
		labels[i] = uint32(Clabel[i])
		dists[i] = float32(Cdist[i])
	}

	return labels[:numResult], dists[:numResult]
}

func (i *Index) SetEfConstruction(efConstruction int) {
	C.setEf(i.index, C.int(efConstruction))
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
