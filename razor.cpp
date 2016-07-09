/*
 * This program is used to compute razor variables from a large dataset of
 * events by taking advantage of parallelization. Event data is read from a file
 * in json format containing, among other things, information about the jets.
 */

#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#include "include/razor.h"

typedef unsigned int uint;

// limit on GPU memory restraining number of tasks to be run in parallel
#define GPUMEM 12000 // currently picked at random
#define GPUCORES 100


/*
 * Input: filename
 * Will attempt to read json format and fill jet vectors
 * Will then run razor variable computation in paralell on GPU
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

    std::cout << "Reading file..." << std::endl;
    // Create array of vectors to store jet data and razor variables later
    std::cout << "Dataset Size: " << obj.size() << std::endl;

    std::vector<Vector> **jet_array = new std::vector<Vector> *[obj.size()];
    Vector *met_array = new Vector[obj.size()];
    double *razor_array = new double[2 * obj.size()];
/*
    double **razor_array = new double *[obj.size()];
    for (int i = 0; i < obj.size(); i++)
        razor_array[i] = new double[2];
*/
    // fill vectors
    std::cout << "Filling vectors in memory with data from file" << std::endl;
    parseJSON(obj, jet_array, met_array);

    // analyzse data in batches in parallel
    std::cout << "Initializing parallel computations" << std::endl;
    runjobs(jet_array, razor_array, met_array, obj.size());
    std::cout << "Finished parallel computations" << std::endl;

    // output razor array results:
    for (int i = 0; i < obj.size(); i += 2)
        std::cout << "MR" << razor_array[i] << "R2" << razor_array[i+1] << std::endl;

    for (int i = 0; i < obj.size(); i++)
        delete jet_array[i];//, razor_array[i];

    delete jet_array, razor_array;
    std::cout  << "Success!" << std::endl;
    return 1;
}



/*
 * Read list of events from json file to and return array of vectors
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
            
            //std::cout << (*jet_iter)["pt"].asDouble() << " " << (*jet_iter)["eta"].asDouble()
            //   << " " << (*jet_iter)["phi"].asDouble() << " " << (*jet_iter)["m"].asDouble() << std::endl;
            
            vec.SetPtEtaPhi(
                    (*jet_iter)["pt"].asDouble() ,
                    (*jet_iter)["eta"].asDouble(),
                    (*jet_iter)["phi"].asDouble());
            jet_array[index]->push_back(vec);
        }

        // obtain MET from event
        met_array[index].SetPtEtaPhi(
                    (*iter)["pt"].asDouble() , 0,
                    (*iter)["phi"].asDouble());
    }
}


/*
 * Run data analysis in parallel in batches, based on available GPU memory
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
        int add_operations = pow(2, njets); // +1 because 2 hemispheres

        // overall memory to be used by this batch of vectors, pick max
        // see runbatch function: batch # * combination * result * vector arrays
        int memory_use = (operations + add_operations) * (
                    sizeof(uint) + 3 * sizeof(double) + sizeof(Vector)
                    * (1 + (max_njets < njets ? njets : max_njets)));

        // If there is enough memory left, add vector to batch and continue
        if (memory_use < GPUMEM && operations < GPUCORES) {

            operations += add_operations;

            if (max_njets < njets)
                max_njets = njets;
        }

        // Otherwise, run this batch, and restart amalgamating
        else {

            // run computation on these -- nb: 2nd index not inclusive
            std::cout << "Processing: " << batch_begin << " - " << iter << " ; " << memory_use << std::endl;
            runbatch(jet_array, razor_array, met_array, batch_begin, iter, max_njets, operations);

            // reset parameters
            operations = add_operations;
            max_njets = njets;
            batch_begin = iter++;
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

    int nevents = ind2 - ind1 - 1;

    // Create a contiguous (pseudo 2D) array of Lorentz Vectors
    Vector *lv_array = new Vector[batch_size * max_size]; // initialized as (0,0,0,0)

    // Create an array to keep all the MET copies
    Vector *mets = new Vector[batch_size];

    // Create an array of all possible combinations of bit vectors of length max_size
    // bit vectors are just going to be implemented as the bits in an integer
    uint *combinations = new uint[batch_size];

    // Store size of the vector for each event for convenience
    int *event_batch_sizes = new int[nevents];
    for (int i = 0; i < nevents; i++)
        event_batch_sizes[i] = jet_array[i]->size();

    // Some indexing ninjutsu to fill these arrays
    uint j = 0;
    for (uint i = 0; i < batch_size; i++) {                         // loop through all processes
        j += !!(i / uint(pow(event_batch_sizes[ind1 + i], 2)));     // keep track of events
        for (uint k = 0; k < event_batch_sizes[ind1 + i]; k++) {    // loop through vectors
            lv_array[i*max_size + k] = jet_array[ind1 + j]->at(k);
            combinations[i*max_size + k] = k;
            mets[i*max_size + k] = met_array[ind1 + j];
        }
    }

    // move these two arrays to device memory
    Vector *dev_lv_array;
    Vector *dev_mets;
    uint *dev_combinations;
    double *dev_results;

    cudaMalloc((void **) &dev_lv_array, batch_size * max_size * sizeof(Vector));
    cudaMemcpy(lv_array, dev_lv_array, batch_size * max_size * sizeof(Vector),
        cudaMemcpyHostToDevice);

    cudaMalloc((void **) &dev_combinations, batch_size * sizeof(uint));
    cudaMemcpy(combinations, dev_combinations, batch_size * sizeof(uint), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &dev_mets, batch_size * sizeof(Vector));
    cudaMemcpy(mets, dev_mets, batch_size * sizeof(Vector), cudaMemcpyHostToDevice);
    
    cudaMalloc((void **) &dev_lv_array, 3 * batch_size * sizeof(double));

    delete lv_array, combinations, mets;

    // Run kernel and obtain jet razor variables and mass information for each process
    // each process is going to be one possible jet partitioning
    run_parallel_jobs(dev_lv_array, dev_combinations, dev_mets, dev_results, batch_size, max_size);

    cudaFree(dev_lv_array);
    cudaFree(dev_combinations);
    cudaFree(dev_mets);

    // Now that this is done we use the gpu to find the optima;
    // dev_results still in gpu memory

    // move size info to device memory
    int *dev_event_batch_sizes;
    cudaMalloc((void **) &dev_event_batch_sizes, nevents * sizeof(int));
    cudaMemcpy(event_batch_sizes, dev_event_batch_sizes, nevents * sizeof(int), cudaMemcpyHostToDevice);

    delete event_batch_sizes;

    // new array to store optimal results
    double *dev_opt_results;
    cudaMalloc((void **) &dev_opt_results, 2 * nevents * sizeof(double));

    // Run second kernel to find optimal razor variables
    find_optimum(dev_results, dev_event_batch_sizes, dev_opt_results, nevents);

    // Copy results to host
    cudaMemcpy(dev_opt_results, razor_array + ind1, 2 * nevents * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_opt_results);
}
