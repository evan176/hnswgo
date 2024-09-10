package hnswgo

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNormalize(t *testing.T) {
	vector := []float32{4, 5, 6}
	normalize(vector)
	var magnitude float32
	for i := range vector {
		magnitude += vector[i] * vector[i]
	}
	magnitude = float32(math.Sqrt(float64(magnitude)))
	assert.Equal(t, magnitude, float32(1.0))
}
