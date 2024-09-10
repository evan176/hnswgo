// hnsw_wrapper.h
#ifdef __cplusplus
extern "C" {
#endif
    void initHNSW(int dim, unsigned long int maxElements, int M, int efConstruction, int randSeed, char simMetric);
    //HNSW loadHNSW(char *location, int dim, char stype);
    //HNSW saveHNSW(HNSW index, char *location);
    void freeHNSW();
    void addPoint(float *vector, unsigned long int label);
    int searchKNN(float *vector, int k, unsigned long int *labels, float *distances);
    void setEf(int ef);
#ifdef __cplusplus
}
#endif
