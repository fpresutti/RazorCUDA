#include <cuda_runtime.h>
#include "include/razor_cuda.cuh"
#include "math_functions.h"

//#define BLOCK_COUNT         1
//#define BLOCK_PER_THREAD    1

/*
 * Redefining vector class
 */
class Vector {
    private:
        double px;
        double py;
        double pz;
    public:
        __device__ double Px() {return px;};
        __device__ double Py() {return py;};
        __device__ double Pz() {return pz;};
        __device__ double Pt() {return sqrt(px*px + py*py);};
        __device__ double P()  {return sqrt(px*px + py*py + pz*pz);};
        __device__ double P2() {return px*px + py*py + pz*pz;};
        __device__ Vector () {
            px = 0;
            py = 0;
            pz = 0;
        }
        __device__ Vector(double a, double b, double c) {
            px = a;
            py = b;
            pz = c;
        };
        __device__ void operator+=(Vector other) {
            px += other.Px();
            py += other.Py();
            pz += other.Pz();
        };
        __device__ Vector operator*(double scale) {
            Vector result(scale * px, scale * py, scale * pz);
            return result;
        };
};

/*
 * Compute the two razor variables, M_R and R^2, and the sum of p magnitude^2
 */
 __device__
void compute_razor(Vector hem1, Vector hem2,
        Vector pfMet, double *result) {
    double mR = sqrt(pow(hem1.P() + hem2.P(), 2) - pow(hem1.Pz() + hem2.Pz(), 2));;
    double term1 = pfMet.Pt() / 2 * (hem1.Pt() + hem2.Pt());
    double term2 = pfMet.Px() / 2 * (hem1.Px() + hem2.Px()) + pfMet.Py() / 2 * (hem1.Py() + hem2.Py());
    double mTR = sqrt(term1 - term2);
    result[0] = mR;
    result[1] = (mTR / mR) * (mTR / mR);
    result[2] = hem1.P2() + hem2.P2();
}

/*
 * GPU Kernels
 * razor_kernel: compute razor variables for all blocks
 * pick_kernel:  iterate through these results and pick result based on mass
 */
__global__ void razor_kernel(Vector *lv_array, uint *combinations,
                  Vector *mets, double *results, int vector_size) {
    Vector hem1, hem2;
    uint bitvector = combinations[blockIdx.x];
    for (int i = 0; i < vector_size; i++) {
        hem1 += (lv_array[blockIdx.x*vector_size + i] * (0x1 & bitvector) );
        hem2 += (lv_array[blockIdx.x*vector_size + i] * (0x1 & ~bitvector));
        bitvector >>= 1;
    }
    compute_razor(hem1, hem2, mets[blockIdx.x], &results[3*blockIdx.x]);
}

__global__ void pick_kernel(double *all_results, int *event_sizes, double *opt_results) {
    double minM = 9e300; // big number
    int min_ind;
    int i = 0;

    int start = 0;
    while (i < blockIdx.x)
        start += event_sizes[i];

    i = 0;
    while (i < event_sizes[blockIdx.x]) {
        // hacky but to avoid divergence
        minM = ( all_results[start+i+2] < minM ? all_results[start+i+2] : minM);
        min_ind = (minM == all_results[start+i+2] ? i : min_ind);
    }
    opt_results[2*blockIdx.x]   = all_results[start + min_ind];
    opt_results[2*blockIdx.x+1] = all_results[start + min_ind + 1];
}

/*
 * Wrapper functions
 */
void run_parallel_jobs(Vector *lv_array, uint *combinations,
        Vector *mets, double *results, int batch_size, int vector_size) {
    razor_kernel<<< batch_size, 1 >>>(lv_array, combinations, mets, results, vector_size);
}

void find_optimum(double *all_results, int *event_batch_sizes,
        double *opt_results, int nevents) {
    pick_kernel<<< nevents, 1 >>>(all_results, event_batch_sizes, opt_results);
}
