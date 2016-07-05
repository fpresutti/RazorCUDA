#include "json.h"

void parseJSON(Json::Value &obj, std::vector<TLorentzVector> **jet_array,
        TLorentzVector *met_array);

void runjobs(std::vector<TLorentzVector> **jet_array, double *razor_array,
        TLorentzVector *met_array, int tot_size);

void runbatch(std::vector<TLorentzVector> **jet_array, double *razor_array,
        TLorentzVector *met_array, int ind1, int ind2, int max_size, int batch_size);
