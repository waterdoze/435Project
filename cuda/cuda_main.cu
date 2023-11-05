#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <ctime>
#include "../common.h"
#include "./cuda_naive.h"
#include "./cuda_strass.h"


#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;



void fillMatrix(int n, int*& mat)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i * n + j] = rand() % 5;
        }
    }
}

int main(int argc, char *argv[]) {
    /*

    Assuming the vectors are created above
    */
    THREADS = atoi(argv[1]);
    MATRIX_SIZE = atoi(argv[2]);
    //Does it make sense to create a thread for every row/column?

    //So we know we have threads amount of threads so can we calculate how blocks we need then?

    BLOCKS = MATRIX_SIZE * MATRIX_SIZE / THREADS;

    

    //When we pass in the thread count it's talking about
    /*
    Issues that it's threads per block
    Essentially each block will take in threads amount of rows and cols and then send them to each block
    thus there will be (n*n)/threads blocks with thread threads inside of it

    steps for cuda naive
    make the array 1d(so change the way we populate)
    when we send the array we only have to send over once and then use the blocking to index the correct ones
    */
    
    



    adiak::init(NULL);
    adiak::launchdate();                                         // launch date of the job
    adiak::libraries();                                          // Libraries used
    adiak::cmdline();                                            // Command line used to launch the job
    adiak::clustername();                                        // Name of the cluster
    adiak::value("Algorithm", "MPI Naive Matrix Multiplication");// The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");          // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", int);                          // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));              // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", N);                        // The number of elements in input dataset (1000)
    // adiak::value("InputType", inputType);                        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", size);                        // The number of processors (MPI ranks)
    // adiak::value("num_threads", num_threads);                    // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", num_blocks);                      // The number of CUDA blocks
    adiak::value("group_num", 8);                     // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online");
    return 0;
}