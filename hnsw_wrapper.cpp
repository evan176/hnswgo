#include "hnswlib/hnswlib.h"
#include "hnsw_wrapper.h"
using namespace std;
using namespace hnswlib;

HNSW initHNSW(int dim, unsigned long int maxElements, int m, int efConstruction, int randSeed, char simMetric) {
    SpaceInterface<float> *vectorSpace;
    if (simMetric == 'i') { // inner product
        vectorSpace = new InnerProductSpace(dim);
    }
    else if (simMetric == 'c') { // cosine (cosine is the same as IP when all vectors are normalized)
        //normalizeVectors = true;
        //spaceDimensions = dim;
        vectorSpace = new InnerProductSpace(dim);
    } else { // default: L2
        vectorSpace = new L2Space(dim);
    }
    return new HierarchicalNSW<float>(vectorSpace, maxElements, m, efConstruction, randSeed); // instantiate the hnsw index
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

void freeHNSW(HNSW hnswIndex) {
    delete (HierarchicalNSW<float>*) hnswIndex;
}

void addPoint(HNSW hnswIndex, float *vector, unsigned long int label) {
    //if (normalizeVectors)
    //    normalizeVec(vector);

    ((HierarchicalNSW<float>*) hnswIndex)->addPoint(vector, label);
}

int searchKNN(HNSW hnswIndex, float *vector, int k, unsigned long int *labels, float *distances) {
    //if (normalizeVectors)
    //    normalizeVec(vector);
    
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

void setEf(HNSW hnswIndex, int ef) {
    ((HierarchicalNSW<float>*) hnswIndex)->ef_ = ef;
}
