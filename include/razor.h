// to benchmark against CPU use
#define CPU


#include "json.h"
#include "vector.h"

#ifndef CPU
#include "razor_cuda.cuh"
#endif

void parseJSON(Json::Value &obj, std::vector<Vector> **jet_array,
        Vector *met_array);

void runjobs(std::vector<Vector> **jet_array, double *razor_array,
        Vector *met_array, int tot_size);

#ifndef CPU
void runbatch(std::vector<Vector> **jet_array, double *razor_array,
        Vector *met_array, int ind1, int ind2, int max_size, int batch_size);
#endif