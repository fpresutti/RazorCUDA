#include <cuda_runtime.h>
#include "include/razor_cuda.cuh"

#define BLOCK_COUNT         1
#define BLOCK_PER_THREAD    1

/*
 * GPU Kernel
 */
__global__
void kernel() {

}

/*
 * Wrapper function
 */
void mykernel() {
    kernel<<<BLOCK_COUNT, BLOCK_PER_THREAD>>>();
}


/*
 * Compute the two razor variables, M_R and R^2
 */
void compute_razor(TLorentzVector hem1, TLorentzVector hem2, TLorentzVector pfMet, double *result) {
    double mR = sqrt(pow(hem1.P() + hem2.P(), 2) - pow(hem1.Pz() + hem2.Pz(), 2));;
    double term1 = pfMet.Pt() / 2 * (hem1.Pt() + hem2.Pt());
    // dot product of MET with (p1T + p2T):
    double term2 = pfMet.Px() / 2 * (hem1.Px() + hem2.Px()) + pfMet.Py() / 2 * (hem1.Py() + hem2.Py());
    double mTR = sqrt(term1 - term2);
    result[0] = mR;
    result[1] = (mTR / mR) * (mTR / mR);
}
