#include "hnswlib/hnswlib.h"
#include "hnsw_wrapper.h"
using namespace std;
using namespace hnswlib;

/**
 * Instantiates and returns an HNSW index.
 *
 * @param dim:              dimension of the vector space
 * @param maxElements:      index's vector storage capacity
 * @param m:                `m` parameter in the HNSW algorithm
 * @param efConstruction:   `efConstruction` parameter in the HNSW algorithm
 * @param randSeed:         random seed
 * @param spaceType:        similarity metric to use in the index
 * 
 * @return                  instance of a HNSW index
 */
HNSW initHNSW(int dim, unsigned long int maxElements, int m, int efConstruction, int randSeed, char spaceType) {
    SpaceInterface<float> *vectorSpace;
    if (spaceType == 'i') { // inner product
        vectorSpace = new InnerProductSpace(dim);
    }
    else if (spaceType == 'c') { // cosine (cosine is the same as IP when all vectors are normalized)
        vectorSpace = new InnerProductSpace(dim);
    } else { // default: L2
        vectorSpace = new L2Space(dim);
    }
    return new HierarchicalNSW<float>(vectorSpace, maxElements, m, efConstruction, randSeed); // instantiate the hnsw index
}

/**
 * Frees an HNSW index from memory.
 *
 * @param hnswIndex: HNSW index to free
 */
void freeHNSW(HNSW hnswIndex) {
    delete (HierarchicalNSW<float>*) hnswIndex;
}

/**
 * Adds a vector to the HNSW index.
 *
 * @param hnswIndex:    HNSW index to add the point to
 * @param vector:       the vector to add to the index
 * @param label:        the vector's label
 */
void insertVector(HNSW hnswIndex, float *vector, unsigned long int label) {
    ((HierarchicalNSW<float>*) hnswIndex)->addPoint(vector, label);
}

/**
 * Performs similarity search on the HNSW index.
 * 
 * @param hnswIndex:    the HNSW index
 * @param vector:       the query vector
 * @param k:            the k value
 * @param labels:       a dynamic array which will receive the labels of the k-nearest neighbors
 * @param distances:    a dynamic array which will receive the distances of the k-nearest neighbors from the query vector
 * 
 * @return              the number of nearest neighbors found (num of nn <= k since it's possible for k > num of vectors in the index)
 */
int searchKNN(HNSW hnswIndex, float *vector, int k, unsigned long int *labels, float *distances) {
    priority_queue<pair<float, labeltype>> searchResults;
    try {
        searchResults = ((HierarchicalNSW<float>*) hnswIndex)->searchKnn(vector, k);
    } catch (const exception e) { 
        return 0; // get better error visibility
    }

    int n = searchResults.size();
    pair<float, labeltype> pair;
    for (int i = n - 1; i >= 0; i--) {
        pair = searchResults.top();
        distances[i] = pair.first;
        labels[i] = pair.second;
        searchResults.pop();
    }
    return n;
}

/**
 * Set's the efConstruction parameter in the HNSW index.
 * 
 * @param hnswIndex:    the HNSW index
 * @param ef:           the new efConstruction parameter
 */
void setEf(HNSW hnswIndex, int ef) {
    ((HierarchicalNSW<float>*) hnswIndex)->ef_ = ef;
}






//HNSW loadHNSW(char *location, int dim, char stype) {
//  SpaceInterface<float> *space;
//  if (stype == 'i') {
//    space = new InnerProductSpace(dim);
//  } else {
//    space = new L2Space(dim);
//  }
//  HierarchicalNSW<float> *appr_alg = new HierarchicalNSW<float>(space, string(location), false, 0);
//  return (void*)appr_alg;
//}
//
//HNSW saveHNSW(HNSW index, char *location) {
//  ((HierarchicalNSW<float>*)index)->saveIndex(location);
//  return ((HierarchicalNSW<float>*)index);
//}