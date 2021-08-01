# HNSWGO
This is a golang interface of [hnswlib](https://github.com/nmslib/hnswlib). For more information, please follow [hnswlib](https://github.com/nmslib/hnswlib) and [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.](https://arxiv.org/abs/1603.09320).

# Compile (Optional)
```bash
git clone github.com/evan176/hnswgo
cd hnswgo && make
sudo cp libhnsw.so /usr/local/lib
ldconfig
```
# Usages
## Download shared library
```bash
sudo wget https://github.com/evan176/hnswgo/releases/download/v1/libhnsw.so -P /usr/local/lib/
ldconfig
```
## Export CGO variable
```
export CGO_CXXFLAGS=-std=c++11
```
## Go get
```
go get github.com/evan176/hnswgo
```

| argument       | type | |
| -------------- | ---- | ----- |
| dim            | int  | vector dimension |
| M              | int  | see[ALGO_PARAMS.md](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) |
| efConstruction | int  | see[ALGO_PARAMS.md](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) |
| randomSeed     | int  | random seed for hnsw |
| maxElements    | int  | max records in data |
| spaceType      | str  | |

| spaceType | distance          |
| --------- |:-----------------:|
| ip        | inner product     |
| cosine    | cosine similarity |
| l2        | l2                |

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
        // randomSeed int = 100
        // Init new index with 1000 vectors in l2 space
	h := hnswgo.New(dim, M, efConstruction, randomSeed, maxElements, spaceType)

        // Insert 1000 vectors to index. Label type is uint32.
	var i uint32
	for ; i < maxElements; i++ {
		h.AddPoint(randVector(dim), i)
	}
	h.Save(indexLocation)
	h = hnswgo.Load(indexLocation, dim, spaceType)

        // Search vector with maximum 10 nearest neighbors
        h.setEf(15)
	searchVector := randVector(dim)
	labels, dists := h.SearchKNN(searchVector, 10)
	for i, l := range labels {
		fmt.Printf("Nearest label: %d, dist: %f\n", l, dists[i])
	}
}
```

# References
Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." TPAMI, preprint: [https://arxiv.org/abs/1603.09320]
