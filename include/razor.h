#include "json.h"

void parseJSON(std::vector<TLorentzVector> **jet_array, Json::Value &obj);
void runjobs(std::vector<TLorentzVector> **jet_array, double **razor_array, int tot_size);
void runbatch(std::vector<TLorentzVector> **jet_array, double **razor_array,
              int ind1, int ind2, int max_size);
