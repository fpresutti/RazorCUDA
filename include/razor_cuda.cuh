
class Vector;

void run_parallel_jobs(Vector *lv_array, uint *combinations,
        Vector *mets, double *results, int batch_size, int vector_size);

void find_optimum(double *all_results, int *event_batch_sizes,
        double *opt_results, int nevents);

