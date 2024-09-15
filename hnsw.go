package hnswgo

/*
#cgo CXXFLAGS: -std=c++11
#include <stdlib.h>
#include <hnsw_wrapper.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"math"
	"unsafe"
)

type Index struct {
	index      C.HNSW
	dimensions int
	size       uint32
	normalize  bool
	spaceType  string
}

// Returns the last error message. Returns nil if there is no error message.
func peekLastError() error {
	err := C.peekLastErrorMsg()
	if err == nil {
		return nil
	}
	return errors.New(C.GoString(err))
}

// Returns and clears the last error message. Returns nil if there is no error message.
func getLastError() error {
	err := C.getLastErrorMsg()
	if err == nil {
		return nil
	}
	return errors.New(C.GoString(err))
}

/*
Normalizes a vector in place.
Normalize(v) = (1/|v|)*v

- vector: the vector to Normalize in place
*/
func Normalize(vector []float32) {
	var magnitude float32
	for i := range vector {
		magnitude += vector[i] * vector[i]
	}
	magnitude = float32(math.Sqrt(float64(magnitude)))

	for i := range vector {
		vector[i] *= 1.0 / magnitude
	}
}

/*
Returns a reference to an instance of an HNSW index.

- dim:            	dimension of the vector space

- maxElements:    	index's vector storage capacity

- m:              	`m` parameter in the HNSW algorithm

- efConstruction: 	`efConstruction` parameter in the HNSW algorithm

- randSeed:       	random seed

- spaceType:      	similarity metric to use in the index

Returns an instance of an HNSW index, or an error if there was a problem initializing the index.
*/
func New(dim int, m int, efConstruction int, randSeed int, maxElements uint32, spaceType string) (*Index, error) {
	if dim < 1 {
		return nil, errors.New("dimension must be >= 1")
	}
	if maxElements < 1 {
		return nil, errors.New("max elements must be >= 1")
	}
	if m < 2 {
		return nil, errors.New("m must be >= 2")
	}
	if efConstruction < 0 {
		return nil, errors.New("efConstruction must be >= 0")
	}

	index := new(Index)
	index.dimensions = dim
	index.spaceType = spaceType
	index.size = maxElements

	if spaceType == "ip" {
		index.index = C.initHNSW(C.int(dim), C.ulong(maxElements), C.int(m), C.int(efConstruction), C.int(randSeed), C.char('i'))
	} else if spaceType == "cosine" {
		index.normalize = true
		index.index = C.initHNSW(C.int(dim), C.ulong(maxElements), C.int(m), C.int(efConstruction), C.int(randSeed), C.char('c'))
	} else {
		index.index = C.initHNSW(C.int(dim), C.ulong(maxElements), C.int(m), C.int(efConstruction), C.int(randSeed), C.char('l'))
	}

	if index.index == nil {
		return nil, getLastError()
	}

	return index, getLastError()
}

/*
Frees the HNSW index from memory.
*/
func (i *Index) Free() {
	C.freeHNSW(i.index)
}

/*
Adds a vector to the HNSW index.

- vector:       the vector to add to the index

- label:        the vector's label
*/
func (i *Index) InsertVector(vector []float32, label uint32) error {
	if len(vector) != i.dimensions {
		return fmt.Errorf("the vector you are trying to insert is %d-dimensional whereas your index is %d-dimensional", len(vector), i.dimensions)
	}

	if i.normalize {
		Normalize(vector)
	}
	C.insertVector(i.index, (*C.float)(unsafe.Pointer(&vector[0])), C.ulong(label))
	return getLastError()
}

/*
Performs similarity search on the HNSW index.

- vector:       the query vector

- k:            the k value

Returns the labels and distances of each of the nearest neighbors. Note: the size of both arrays can be < k if k > num of vectors in the index
*/
func (i *Index) SearchKNN(vector []float32, k int) ([]uint32, []float32, error) {
	if len(vector) != i.dimensions {
		return nil, nil, fmt.Errorf("the query vector is %d-dimensional whereas your index is %d-dimensional", len(vector), i.dimensions)
	}
	if k < 1 || uint32(k) > i.size {
		return nil, nil, fmt.Errorf("1 <= k <= index max size")
	}

	if i.normalize {
		Normalize(vector)
	}

	Clabel := make([]C.ulong, k, k)
	Cdist := make([]C.float, k, k)

	numResult := int(C.searchKNN(i.index, (*C.float)(unsafe.Pointer(&vector[0])), C.int(k), &Clabel[0], &Cdist[0])) // perform the search

	if numResult < 0 {
		return nil, nil, fmt.Errorf("an error occured with the HNSW algorithm: %s", getLastError())
	}

	labels := make([]uint32, k)
	dists := make([]float32, k)
	for i := 0; i < numResult; i++ {
		labels[i] = uint32(Clabel[i])
		dists[i] = float32(Cdist[i])
	}

	return labels[:numResult], dists[:numResult], getLastError()
}

/*
Set's the efConstruction parameter in the HNSW index.

- efConstruction: the new efConstruction parameter
*/
func (i *Index) SetEfConstruction(efConstruction int) error {
	if efConstruction < 0 {
		return errors.New("efConstruction must be >= 0")
	}
	C.setEf(i.index, C.int(efConstruction))
	return getLastError()
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
