#include <TLorentzVector.h>

void run_parallel_jobs(TLorentzVector *lv_array, uint *combinations,
        TLorentzVector *mets, double *results, int batch_size, int vector_size);

void compute_razor(TLorentzVector hem1, TLorentzVector hem2,
        TLorentzVector pfMet, double *result, int vector_size);

void find_optimum(double *all_results, int *event_batch_sizes,
        double *opt_results, int nevents);
