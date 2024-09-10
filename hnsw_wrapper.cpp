#include <hnswlib/hnswlib.h>
#include <hnsw_wrapper.h>
#include <math.h>
#include <stdlib.h>
using namespace std;
using namespace hnswlib;

SpaceInterface<float> *vectorSpace;
HierarchicalNSW<float> *hnswIndex;
bool normalizeVectors = false;
int spaceDimensions;

void normalizeVec(float *vector) { // maybe vector normalization should be outsourced to Go? It's safer.
    float magnitude = 0.0f;
    for (int i = 0; i < spaceDimensions; i++) {
        magnitude += vector[i] * vector[i];
    }
    magnitude = sqrt(magnitude);

    for (int i = 0; i < spaceDimensions; i++) {
        vector[i] *= 1 / magnitude;
    }
}

void initHNSW(int dim, unsigned long int maxElements, int m, int efConstruction, int randSeed, char simMetric) {
    if (simMetric == 'i') { // inner product
        vectorSpace = new InnerProductSpace(dim);
    }
    else if (simMetric == 'c') { // cosine (cosine is the same as IP when all vectors are normalized)
        normalizeVectors = true;
        spaceDimensions = dim;
        vectorSpace = new InnerProductSpace(dim);
    } else { // default: L2
        vectorSpace = new L2Space(dim);
    }
    hnswIndex = new HierarchicalNSW<float>(vectorSpace, maxElements, m, efConstruction, randSeed); // instantiate the hnsw index
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

void freeHNSW() {
    delete hnswIndex;
}

void addPoint(float *vector, unsigned long int label) {
    if (normalizeVectors)
        normalizeVec(vector);

    hnswIndex->addPoint(vector, label);
}

int searchKNN(float *vector, int k, unsigned long int *labels, float *distances) {
    if (normalizeVectors)
        normalizeVec(vector);
    
    priority_queue<pair<float, labeltype>> searchResults;
    try {
        searchResults = hnswIndex->searchKnn(vector, k);
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

void setEf(int ef) {
    hnswIndex->ef_ = ef;
}
