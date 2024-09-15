# HNSWGO

### A Go wrapper for [hnswlib](https://github.com/nmslib/hnswlib) 📦 

## Installation

test

```
go get github.com/Eigen-DB/hnswgo
```

## Usage

```go
package examples

import (
	"fmt"
	"time"

	"github.com/Eigen-DB/hnswgo"
)

func main() {
	dimensions := 2
	maxElements := 10000
	m := 32
	efConstruction := 400
	spaceType := "l2"
	seed := int(time.Now().Unix())

	// instantiate the index
	index, err := hnswgo.New(
		dimensions,
		m,
		efConstruction,
		seed,
		uint32(maxElements),
		spaceType,
	)
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}

	defer index.Free() // defer freeing the index from memory (don't forget in order ot prevent memory leaks)

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
		err = index.InsertVector(v, uint32(i))
		if err != nil {
			fmt.Printf("Error: %s\n", err.Error())
		}
	}

	k := 5
	nnLabels, nnDists, err := index.SearchKNN(vectors[0], k) // perform similarity search where the first of our sample vectors is the query vector
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}

	fmt.Printf("%d-nearest neighbors:\n", k)
	for i := range nnLabels {
		fmt.Printf("vector %d is %f units from query vector\n", nnLabels[i], nnDists[i])
	}
}
```

Visualize the vectors in this example [here](https://www.desmos.com/calculator/n47sh892rk).
