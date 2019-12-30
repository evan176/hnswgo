# HNSWGO
This is a golang interface of [hnswlib](https://github.com/nmslib/hnswlib). For more information, please follow [hnswlib](https://github.com/nmslib/hnswlib) and [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.](https://arxiv.org/abs/1603.09320).

# Compile
```bash
go get github.com/evan176/hnswgo
cd go/src/github.com/evan176/hnswgo && make
```
# Usages
Specify environment before compiling golang
```bash
export CGO_CXXFLAGS=-std=c++11
```

```go
package main

import (
	"fmt"
	"math/rand"

	"github.com/evan176/hnswgo"
)

func randVector(dim int) []float32 {
	vec := make([]float32, dim)
	for j := 0; j < dim; j++ {
		vec[j] = rand.Float32()
	}
	return vec
}

func main() {
	var dim, M, efConstruction int = 128, 32, 300
	// Maximum elements need to construct index
	var maxElements uint32 = 1000
	// Define search space: l2 or ip (innder product)
	var spaceType, indexLocation string = "l2", "hnsw_index.bin"
        // Init new index with 1000 vectors in l2 space
	h := hnswgo.New(dim, M, efConstruction, 0, maxElements, spaceType)

        // Insert 1000 vectors to index. Label type is uint32.
	var i uint32
	for ; i < maxElements; i++ {
		h.AddPoint(randVector(dim), i)
	}
	h.Save(indexLocation)
	h = hnswgo.Load(indexLocation, dim, spaceType)

        // Search vector with maximum 10 nearest neighbors
	searchVector := randVector(dim)
	labels := h.SearchKNN(searchVector, 10)
	for _, l := range labels {
		fmt.Printf("Nearest label: %d\n", l)
	}
}
```

# References
Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." TPAMI, preprint: [https://arxiv.org/abs/1603.09320]
