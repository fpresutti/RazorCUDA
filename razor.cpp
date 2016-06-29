/*
 * This program is used to compute razor variables from a large dataset of
 * events by taking advantage of parallelization. Event data is read from a file
 * in json format containing, among other things, information about the jets.
 */

#include <iostream>
#include <fstream>
#include "TLorentzVector.h"

#include "include/razor.h"
#include "include/json.h"
//#include "include/razor_cuda.cuh"


// limit on GPU memory restraining number of tasks to be run in parallel
#define GPUMEM 12000
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

    // Create array of vectors to store jet data and razor variables later
    std::vector<TLorentzVector> **jet_array = new std::vector<TLorentzVector> *[obj.size()];
    double **razor_array = new double *[obj.size()];
    for (int i = 0; i < obj.size(); i++)
        razor_array[i] = new double[2];
    std::cout << "Size: " << obj.size() << std::endl;

    // fill vectors
    parseJSON(jet_array, obj);

    // analyzse data in batches in parallel
    runjobs(jet_array, razor_array, obj.size());


    for (int i = 0; i < obj.size(); i++)
        delete jet_array[i], razor_array[i];

    delete jet_array, razor_array;
    std::cout  << "Success!" << std::endl;
    return 1;
}



/*
 * Read list of events from json file to and return array of vectors
 */
void parseJSON(std::vector<TLorentzVector> **jet_array, Json::Value &obj) {

    // iterate through array of events in json file
    Json::ValueIterator iter;
    int index;
    for (iter = obj.begin(), index = 0; iter != obj.end(); iter++, index++) {
        
        jet_array[index] = new std::vector<TLorentzVector>(0);

        // iterate through all the jets in this particular event
        for (Json::ValueIterator jet_iter = (*iter)["event"]["jets"].begin();
                jet_iter != (*iter)["event"]["jets"].end(); jet_iter++) {
            TLorentzVector vec;
            
            //std::cout << (*jet_iter)["pt"].asDouble() << " " << (*jet_iter)["eta"].asDouble()
            //   << " " << (*jet_iter)["phi"].asDouble() << " " << (*jet_iter)["m"].asDouble() << std::endl;
            
            vec.SetPtEtaPhiM(
                    (*jet_iter)["pt"].asDouble() ,
                    (*jet_iter)["eta"].asDouble(),
                    (*jet_iter)["phi"].asDouble(),
                    (*jet_iter)["m"].asDouble() );
            jet_array[index]->push_back(vec);
        }
    }
}


/*
 * Run data analysis in parallel in batches based on available GPU memory
 */
void runjobs(std::vector<TLorentzVector> **jet_array, double **razor_array, int tot_size) {

    // maximum number of jets in this batch of vectors
    int max_njets = 0;
    // operations to be performed in a batch (includes all combinations)
    int operations = 0;
    // index of first vector in batch
    int batch_begin = 0;

    for (int iter = 0; iter < tot_size; iter++) {
        std::cout << iter << std::endl;

        // number of jets in this vector
        int njets = jet_array[iter]->size();

        // additional operations needed to be performed with this vector
        int add_operations = pow(2, njets + 1); // +1 because 2 hemispheres

        // overall memory to be used by this batch of vectors, pick max
        int memory_use = ( max_njets < njets ?
            operations + add_operations * njets*njets * 100//sizeof(something)
            :
            operations + add_operations * max_njets*max_njets * 100//sizeof(something)
            );

        // If there is enough memory left, add vector to batch and continue
        if (memory_use < GPUMEM && operations < GPUCORES) {

            operations += add_operations;

            if (max_njets < njets)
                max_njets = njets;
        }

        // Otherwise, run this batch, and restart amalgamating
        else {

            // run computation on these -- nb: 2nd index not inclusive
            runbatch(jet_array, razor_array, batch_begin, iter, max_njets);
            std::cout << batch_begin << " - " << iter << " ; " << memory_use << std::endl;

            // reset parameters
            operations = add_operations;
            max_njets = njets;
            batch_begin = iter++;
        }
    }
    // run last batch
    runbatch(jet_array, razor_array, batch_begin, tot_size, max_njets);

}


/*
 * Run a given batch of vectors on the GPU
 * First create memory arrays containing bit vectors and Lorentz vectors
 * then copy to GPU and run Kernel
 */
void runbatch(std::vector<TLorentzVector> **jet_array, double **razor_array,
              int ind1, int ind2, int max_size) {


}
