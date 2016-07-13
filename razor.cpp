/*
 * This program is used to compute razor variables from a large dataset of
 * events by taking advantage of parallelization. Event data is read from a file
 * in json format that must contain an array of events including jets and met data.
 */

#include <iostream>
#include <fstream>

#ifndef CPU
#include <cuda_runtime.h>
#endif

#include "include/razor.h"

typedef unsigned int uint;

// limit on GPU memory restraining number of tasks to be run in parallel
#define GPUMEM   200000 // 12000000000
#define GPUCORES 200  // 3000
#define GPUNUM   1

// Error checking with CUDA
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*
 * Input: filename string
 *
 * Program will attempt to read json file and fill (C++) vectors of jet Vectors.
 * Will then run razor variable computation in parallel on GPU for all possible
 * combinations of jet partitions.
 * Will then pick partition minimizing partition-hemisphere Vector magnitude^2,
 * and output results.
 */
int main (int argc, char **argv) {

    // Obtain array of vectors of Lorentz vectors
    // One cell per event, vectors are variable length

    // initialize objects for json parsing
    Json::Value obj;
    Json::Reader reader;
    // open file stream for json data file 
    std::ifstream jsondata(argv[1], std::ifstream::binary);
    // report any failure
    if (! reader.parse(jsondata, obj)) {
        std::cout  << "Failed to parse configuration" << std::endl
                   << reader.getFormattedErrorMessages();
        return 0;
    }

    // Create array of vectors to store jet data and razor variables later
    std::cout << "Dataset Size: " << obj.size() << std::endl;

    std::vector<Vector> **jet_array = new std::vector<Vector> *[obj.size()];
    Vector *met_array = new Vector[obj.size()];
    double *razor_array = new double[2 * obj.size()];

    // fill vectors
    std::cout << "Filling vectors in memory with data from file" << std::endl;
    parseJSON(obj, jet_array, met_array);

    // analyzse data in batches in parallel
    std::cout << "Initializing parallel computations" << std::endl;
    runjobs(jet_array, razor_array, met_array, obj.size());
    std::cout << "Finished parallel computations" << std::endl;

    /*
    // output razor array results:
    for (int i = 0; i < 2*obj.size(); i += 2)
        std::cout << "MR: " << razor_array[i] << ", R2: " << razor_array[i+1] << std::endl;
    */
    for (int i = 0; i < obj.size(); i++)
        delete jet_array[i];

    delete jet_array, razor_array, met_array;

    return 1;
}


/*
 * Read list of events from json file to and return array of vectors of jet Vectors
 */
void parseJSON(Json::Value &obj, std::vector<Vector> **jet_array,
               Vector *met_array) {

    // iterate through array of events in json file
    Json::ValueIterator iter;
    int index;
    for (iter = obj.begin(), index = 0; iter != obj.end(); iter++, index++) {
        
        jet_array[index] = new std::vector<Vector>(0);

        // iterate through all the jets in this particular event
        for (Json::ValueIterator jet_iter = (*iter)["event"]["jets"].begin();
                jet_iter != (*iter)["event"]["jets"].end(); jet_iter++) {

            Vector vec;
            
            vec.SetPtEtaPhi(
                    (*jet_iter)["pt"].asDouble() ,
                    (*jet_iter)["eta"].asDouble(),
                    (*jet_iter)["phi"].asDouble());
            jet_array[index]->push_back(vec);
        }

        // obtain MET from event
        met_array[index].SetPtEtaPhi(
                    (*iter)["event"]["met"]["pt"].asDouble(), 0,
                    (*iter)["event"]["met"]["phi"].asDouble());
    }
}

#ifndef CPU

/*
 * Run data analysis in parallel in batches, based on available GPU cores/memory
 */
void runjobs(std::vector<Vector> **jet_array, double *razor_array,
             Vector *met_array, int tot_size) {

    // maximum number of jets in this batch of vectors
    int max_njets = 0;
    // operations to be performed in a batch (includes all combinations)
    int operations = 0;
    // index of first vector in batch
    int batch_begin = 0;

    for (int iter = 0; iter < tot_size; iter++) {
        //std::cout << iter << std::endl;

        // number of jets in this vector
        int njets = jet_array[iter]->size();

        // additional operations needed to be performed with this vector
        int add_operations = pow(2, njets);

        // overall memory to be used by this batch of vectors, pick max
        // see runbatch function: batch # * combination * result * vector arrays
        int memory_use = (operations + add_operations) * (
                    sizeof(uint) + 3 * sizeof(double) + sizeof(Vector)
                    * (1 + (max_njets < njets ? njets : max_njets)));

        // If there is enough memory left, add vector to batch and continue
        if (memory_use < GPUMEM && operations + add_operations < GPUCORES) {

            operations += add_operations;

            if (max_njets < njets)
                max_njets = njets;
        }

        // Otherwise, run this batch, and restart amalgamating
        else {

            // run computation on these -- nb: 2nd index not inclusive
            //std::cout << "Processing: " << batch_begin << " - " << iter << " ; " << operations << std::endl;
            runbatch(jet_array, razor_array, met_array, batch_begin, iter, max_njets, operations);

            // reset parameters
            operations = add_operations;
            max_njets = njets;
            batch_begin = iter;
        }
    }
    // run last batch
    runbatch(jet_array, razor_array, met_array, batch_begin, tot_size, max_njets, operations);

}


/*
 * Run a given batch of vectors on the GPU
 * First create memory arrays containing bit vectors and Lorentz vectors
 * then copy to GPU and run Kernel
 * Once these results are obtained run a second kernel to find the optimal
 * result for each event
 */
void runbatch(std::vector<Vector> **jet_array, double *razor_array,
              Vector *met_array, int ind1, int ind2, int max_size, int batch_size) {

    int nevents = ind2 - ind1;

    // Create a contiguous (pseudo 2D) array of Lorentz Vectors
    Vector *lv_array = new Vector[batch_size * max_size]; // initialized as (0,0,0,0)

    // Create an array to keep all the MET copies
    Vector *mets = new Vector[batch_size];

    // Create an array of all possible combinations of bit vectors of length max_size
    // bit vectors are just going to be implemented as the bits in an integer
    uint *combinations = new uint[batch_size];

    // Store size of the vector for each event for convenience
    int *event_jet_num = new int[nevents];
    for (int i = 0; i < nevents; i++)
        event_jet_num[i] = jet_array[ind1 + i]->size();

    // Some indexing ninjutsu to fill these arrays
    uint j = ind1;
    uint event_tracker = uint(pow(2, event_jet_num[j-ind1]));
    for (uint i = 0; i < batch_size; i++) {                 // loop through all processes
        if (event_tracker - i == 0) {                       // keep track of events
            j++;
            event_tracker += uint(pow(2, event_jet_num[j-ind1]));
        }
        for (uint k = 0; k < event_jet_num[j-ind1]; k++) {  // loop through Vectors
            //std::cout << i << ", " << j << ", " << k << std::endl;
            lv_array[i*max_size + k] = jet_array[j]->at(k);
        }
        combinations[i] = event_tracker - i - 1;
        mets[i] = met_array[j];
    }

    cudaSetDevice(GPUNUM);

    // move these two arrays to device memory
    Vector *dev_lv_array;
    Vector *dev_mets;
    uint *dev_combinations;
    double *dev_results;

    gpuErrchk( cudaMalloc((void **) &dev_lv_array, batch_size * max_size * sizeof(Vector)) );
    gpuErrchk( cudaMemcpy(dev_lv_array, lv_array, batch_size * max_size * sizeof(Vector), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc((void **) &dev_combinations, batch_size * sizeof(uint)) );
    gpuErrchk( cudaMemcpy(dev_combinations, combinations, batch_size * sizeof(uint), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc((void **) &dev_mets, batch_size * sizeof(Vector)) );
    gpuErrchk( cudaMemcpy(dev_mets, mets, batch_size * sizeof(Vector), cudaMemcpyHostToDevice) );
    
    gpuErrchk( cudaMalloc((void **) &dev_results, 3 * batch_size * sizeof(double)) );

    delete lv_array, combinations, mets;

    // Run kernel and obtain jet razor variables and mass information for each process
    // each process is going to be one possible jet partitioning
    gpuErrchk( cudaThreadSynchronize() );
    run_parallel_jobs(dev_lv_array, dev_combinations, dev_mets, dev_results, batch_size, max_size);
    gpuErrchk( cudaThreadSynchronize() );

    gpuErrchk( cudaFree(dev_lv_array) );
    gpuErrchk( cudaFree(dev_combinations) );
    gpuErrchk( cudaFree(dev_mets) );
/*
    // Now that this is done we use the gpu to find the optima;
    // dev_results still in gpu memory

    // move size info to device memory
    int *dev_event_jet_num;
    gpuErrchk( cudaMalloc((void **) &dev_event_jet_num, nevents * sizeof(int)));
    gpuErrchk( cudaMemcpy(dev_event_jet_num, event_jet_num, nevents * sizeof(int), cudaMemcpyHostToDevice));

    delete event_jet_num;

    // new array to store optimal results
    double *dev_opt_results;
    gpuErrchk( cudaMalloc((void **) &dev_opt_results, 2 * nevents * sizeof(double)) );

    // Run second kernel to find optimal razor variables
    std:: cout << "Finding optimal razor variables for each event" << std::endl;
    gpuErrchk( cudaThreadSynchronize() );
    find_optimum(dev_results, dev_event_jet_num, dev_opt_results, nevents);
    gpuErrchk( cudaThreadSynchronize() );
    std::cout << "Done" << std::endl;

    gpuErrchk( cudaFree(dev_event_jet_num) );

    // Copy results to host
    std:: cout << "Copying results to host" << std::endl;
    gpuErrchk( cudaMemcpy(razor_array + ind1, dev_opt_results, 2 * nevents * sizeof(double), cudaMemcpyDeviceToHost) );
    std:: cout << "Done" << std::endl;

    gpuErrchk( cudaFree(dev_opt_results) );
*/

    // Copy results
    double *results = new double[3 * batch_size];
    cudaMemcpy(results, dev_results, 3 * batch_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dev_results);

    // Iterate through results, looking for min magnitude parameter
    double mMin = 9e300; // big ol' number to start with
    j = ind1;
    event_tracker = uint(pow(2, event_jet_num[j-ind1]));
    //std::cout << j << ": (" << event_jet_num[j-ind1] << ") " << std::endl;
    for (uint i = 0; i < batch_size; i++) {
        if (event_tracker - i == 0) { // reset at new j index
            j++;
            event_tracker += uint(pow(2, event_jet_num[j-ind1]));
            mMin = 9e300;
            //std::cout << j << ": (" << event_jet_num[j-ind1] << ") " << std::endl;
        }
        if (results[3*i + 2] < mMin && results[3*i + 2] != 0) { // found lower value
            mMin = results[3*i + 2];
            razor_array[2*j]   = results[3*i];   // obtain m_r
            razor_array[2*j+1] = results[3*i+1]; // obtain r^2
        }
    }

    delete event_jet_num, results;
}

#else

/*
 * Run data analysis serially on CPU
 */
void runjobs(std::vector<Vector> **jet_array, double *razor_array,
             Vector *met_array, int tot_size) {

    for (int event = 0; event < tot_size; event++) {
        // obscure code from razor analysis
        int nComb = pow(2, jet_array[event]->size());
        int j_count;
        double bestmag2 = 9e300;
        Vector best1, best2;
        for (int i = 1; i < nComb - 1; i++) {
            Vector j_temp1, j_temp2;
            int itemp = i;
            j_count = nComb / 2;
            int count = 0;
            while (j_count > 0) {
                if (itemp / j_count == 1) {
                    j_temp1 += jet_array[event]->at(count);
                }
                else {
                    j_temp2 += jet_array[event]->at(count);
                }
                itemp -= j_count * (itemp / j_count);
                j_count /= 2;
                count++;
            }
            // Check mag^2 and see if better new razor variables
            if (j_temp1.P2() + j_temp2.P2() < bestmag2) {
                bestmag2 = best1.P2() + best2.P2();
                best1 = j_temp1;
                best2 = j_temp2;
            }
        }
        // fill results with this vector's crazor variables
        double mR = sqrt(pow(best1.P() + best2.P(), 2) - pow(best1.Pz() + best2.Pz(), 2));;
        double term1 = met_array[event].Pt() / 2 * (best1.Pt() + best2.Pt());
        double term2 = met_array[event].Px() / 2 * (best1.Px() + best2.Px())
                + met_array[event].Py() / 2 * (best1.Py() + best2.Py());
        double mTR = sqrt(term1 - term2);
        razor_array[2*event] = mR;
        razor_array[2*event+1] = (mTR / mR) * (mTR / mR);
    }
}

#endif