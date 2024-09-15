package tests

import (
	"fmt"
	"math"
	"sort"
	"testing"
	"time"

	"github.com/Eigen-DB/hnswgo"

	"github.com/stretchr/testify/assert"
)

func setup() (*hnswgo.Index, error) {
	index, err := hnswgo.New(
		2,
		32,
		400,
		int(time.Now().Unix()),
		uint32(10000),
		"l2",
	)

	if err != nil {
		return nil, fmt.Errorf("An error occured when instantiating the index: %s", err.Error())
	}
	return index, nil
}

func TestNormalize(t *testing.T) {
	vector := []float32{4, 5, 6}
	hnswgo.Normalize(vector)
	var magnitude float32
	for i := range vector {
		magnitude += vector[i] * vector[i]
	}
	magnitude = float32(math.Sqrt(float64(magnitude)))
	assert.Equal(t, magnitude, float32(1.0))
}

func TestNewSuccess(t *testing.T) {
	dim := 2
	maxElements := uint32(10000)
	m := 32
	efConstruction := 400
	spaceType := "l2"
	seed := int(time.Now().Unix())

	index, err := hnswgo.New(
		dim,
		m,
		efConstruction,
		seed,
		maxElements,
		spaceType,
	)

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	} else {
		defer index.Free()
	}

	if index == nil {
		t.Fatal("Expected valid index, got nil")
	}
}

func TestNewFailure(t *testing.T) {
	dim := -128 // Invalid dimension
	m := 16
	efConstruction := 200
	randSeed := int(time.Now().Unix())
	maxElements := uint32(10000)
	spaceType := "l2"

	index, err := hnswgo.New(
		dim,
		m,
		efConstruction,
		randSeed,
		maxElements,
		spaceType,
	)

	if err == nil {
		t.Fatal("Expected an error for invalid parameters, but got none")
	}

	if index != nil {
		defer index.Free()
		t.Fatal("Expected nil index on failure, but got valid index")
	}
}

func TestInsertVectorSuccess(t *testing.T) {
	index, err := setup()
	if err != nil {
		t.Fatal(err.Error())
	}
	defer index.Free()

	vector := []float32{1.2, -4.2}
	err = index.InsertVector(vector, 1)
	if err != nil {
		t.Fatalf("An error occured when inserting a vector: %s", err.Error())
	}
}

func TestInsertVectorFailure(t *testing.T) {
	index, err := setup()
	if err != nil {
		t.Fatal(err.Error())
	}
	defer index.Free()

	vector := []float32{1.2, -4.2, 3.3} // trying to insert 3-dimensional vector in 2-dimensional index -> error
	err = index.InsertVector(vector, 1)
	if err == nil {
		t.Fatal("An error SHOULD HAVE occured when inserting a 3D vector in a 2D index")
	}
}

func TestSearchKNN(t *testing.T) {
	index, err := setup()
	if err != nil {
		t.Fatal(err.Error())
	}
	defer index.Free()

	// sample vectors
	vectors := [][]float32{
		{1.2, 3.4},
		{2.1, 4.5},
		{0.5, 1.7},
		{3.3, 2.2},
		{4.8, 5.6},
		{7.1, 8.2},
		{9.0, 0.4},
		{6.3, 3.5},
		{2.9, 7.8},
		{5.0, 1.1},
	}

	// insert sample vectors
	for i, v := range vectors {
		index.InsertVector(v, uint32(i))
	}

	k := 5
	nnLabels, nnDists, err := index.SearchKNN(vectors[0], k) // perform similarity search where the first of our sample vectors is the query vector
	if err != nil {
		t.Fatalf("Error when performing similarity search: %s", err.Error())
	}

	sort.Slice(nnLabels, func(i, j int) bool {
		return nnLabels[i] < nnLabels[j]
	})

	assert.Equal(t, []uint32{0, 1, 2, 3, 4}, nnLabels)

	t.Logf("%d-nearest neighbors:\n", k)
	for i := range nnLabels {
		t.Logf("vector %d is %f units from query vector\n", nnLabels[i], nnDists[i])
	}
}

func TestSetEfConstructionSuccess(t *testing.T) {
	index, err := setup()
	if err != nil {
		t.Fatal(err.Error())
	}
	defer index.Free()

	err = index.SetEfConstruction(401)
	if err != nil {
		t.Fatalf("An error occured when updating efConstruction: %s", err.Error())
	}
}

func TestSetEfConstructionFailure(t *testing.T) {
	index, err := setup()
	if err != nil {
		t.Fatal(err.Error())
	}
	defer index.Free()

	err = index.SetEfConstruction(-1)
	if err == nil {
		t.Fatal("An error SHOULD HAVE occured when updating efConstruction.")
	}
}
