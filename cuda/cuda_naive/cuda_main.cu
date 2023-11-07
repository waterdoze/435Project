#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <functional>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas.h>


#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;
int MATRIX_SIZE;

__global__ void naive(int* a, int* b,int* c,int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    c[row * N + col] = 0;
    for (int k = 0; k < N; k++) {
            // Accumulate results for a single element
        c[row * N + col] += a[row * N + k] * b[k * N + col];
    }
}
void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}


int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    /*
    Assuming the vectors are created above
    */
    const char* data_init = "data_init";
    const char* comm = "comm";
    const char* comp =  "comp";
    const char* correctness = "correctness";


    const char* host2device = "host2device";
    const char* device2host = "device2host";
    const char* cuda_naive_time = "cuda_naive_time";

    THREADS = atoi(argv[1]);
    MATRIX_SIZE = atoi(argv[2]);
    //Does it make sense to create a thread for every row/column?

    //So we know we have threads amount of threads so can we calculate how blocks we need then?

    BLOCKS = (MATRIX_SIZE + THREADS - 1) / THREADS;
    int h_a[MATRIX_SIZE * MATRIX_SIZE];
    int h_b[MATRIX_SIZE * MATRIX_SIZE];
    int h_c[MATRIX_SIZE * MATRIX_SIZE];

    size_t bytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(int);

    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN(data_init);
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            h_a[i * MATRIX_SIZE + j] = rand() % 10 + 1;
            h_b[i * MATRIX_SIZE + j] = rand() % 10 + 1;
            h_c[i * MATRIX_SIZE + j] = 0;
        }
    }
    CALI_MARK_END(data_init);

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(host2device);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    CALI_MARK_END(host2device);
    CALI_MARK_END(comm);

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    float start = clock();
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(cuda_naive_time);
    naive<<<blocks, threads>>>(d_a, d_b, d_c, MATRIX_SIZE);
    CALI_MARK_END(cuda_naive_time);
    CALI_MARK_END(comp);
    float stop = clock();
    print_elapsed(start,stop);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(device2host);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    CALI_MARK_END(device2host);
    CALI_MARK_END(comm);

    //When we pass in the thread count it's talking about
    /*
    Issues that it's threads per block
    Essentially each block will take in threads amount of rows and cols and then send them to each block
    thus there will be (n*n)/threads blocks with thread threads inside of it

    steps for cuda naive
    make the array 1d(so change the way we populate)
    when we send the array we only have to send over once and then use the blocking to index the correct ones
    */

    // Verify results
    CALI_MARK_BEGIN(correctness);
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            int check = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                check += h_a[i * MATRIX_SIZE + k] * h_b[k * MATRIX_SIZE + j];
            }
            if (check != h_c[i * MATRIX_SIZE + j]) {
                printf("Error: %d != %d\n", check, h_c[i * MATRIX_SIZE + j]);
                exit(1);
            }
        }
    }
    CALI_MARK_END(correctness);


    adiak::init(NULL);
    adiak::launchdate();                                         // launch date of the job
    adiak::libraries();                                          // Libraries used
    adiak::cmdline();                                            // Command line used to launch the job
    adiak::clustername();                                        // Name of the cluster
    adiak::value("Algorithm", "CUDA Naive Matrix Multiplication");// The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA");          // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                          // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));              // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", MATRIX_SIZE);                        // The number of elements in input dataset (1000)
    // adiak::value("InputType", inputType);                        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    //adiak::value("num_procs", size);                        // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS);                    // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS);                      // The number of CUDA blocks
    adiak::value("group_num", 8);                     // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online");

    mgr.stop();
    mgr.flush();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}

