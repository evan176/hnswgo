#ifdef __cplusplus
extern "C" {
#endif
    typedef void* HNSW;
    HNSW initHNSW(int dim, unsigned long int maxElements, int m, int efConstruction, int randSeed, char simMetric);
    //HNSW loadHNSW(char *location, int dim, char stype);
    //HNSW saveHNSW(HNSW index, char *location);
    void freeHNSW(HNSW hnswIndex);
    void addPoint(HNSW hnswIndex, float *vector, unsigned long int label);
    int searchKNN(HNSW hnswIndex, float *vector, int k, unsigned long int *labels, float *distances);
    void setEf(HNSW hnswIndex, int ef);
#ifdef __cplusplus
}
#endif
