//hnsw_wrapper.cpp
#include <hnswlib/hnswlib.h>
#include <hnsw_wrapper.h>
using namespace std;
using namespace hnswlib;

SpaceInterface<float> *vectorSpace;
HierarchicalNSW<float> *hnswIndex;

void initHNSW(int dim, unsigned long int maxElements, int M, int efConstruction, int randSeed, char simMetric) {
    if (simMetric == 'l2') { // L2 (Euclidean)
        vectorSpace = new L2Space(dim);
    } else { // IP or cosine (cosine is the same as IP when all vectors are normalized)
        vectorSpace = new InnerProductSpace(dim);
    }
    hnswIndex = new HierarchicalNSW<float>(vectorSpace, maxElements, M, efConstruction, randSeed);
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
    hnswIndex->addPoint(vector, label);
}

int searchKNN(float *vector, int k, unsigned long int *labels, float *distances) {
    priority_queue<pair<float, labeltype>> searchResults;
    try {
        searchResults = hnswIndex->searchKnn(vector, k);
    } catch (const exception e) { 
        return 1;
    }

    int n = searchResults.size();
    pair<float, labeltype> pair;
    for (int i = n - 1; i >= 0; i--) {
        pair = searchResults.top();
        distances[i] = pair.first;
        labels[i] = pair.second; // can i index a dynamic array ? do i need to use ptr offsets?
        searchResults.pop();
    }
    return n;
}

void setEf(int ef) {
    hnswIndex->ef_ = ef;
}
