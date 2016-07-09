#include "json.h"
#include "vector.h"
#include "razor_cuda.cuh"

void parseJSON(Json::Value &obj, std::vector<Vector> **jet_array,
        Vector *met_array);

void runjobs(std::vector<Vector> **jet_array, double *razor_array,
        Vector *met_array, int tot_size);

void runbatch(std::vector<Vector> **jet_array, double *razor_array,
        Vector *met_array, int ind1, int ind2, int max_size, int batch_size);
