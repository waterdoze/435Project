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

__global__ void naive(int* matrixA, int* matrixB,int* matrixC,int matSize){
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

    const char* data_init = "data_init";
    const char* host2device = "host2device";
    const char* device2host = "device2host";
    const char* naive_time = "naive_time";

    THREADS = atoi(argv[1]);
    MATRIX_SIZE = atoi(argv[2]);
    //Does it make sense to create a thread for every row/column?

    //So we know we have threads amount of threads so can we calculate how blocks we need then?

    BLOCKS = (MATRIX_SIZE + THREADS - 1) / THREADS;

    vector<int> h_a(MATRIX_SIZE * MATRIX_SIZE);
    vector<int> h_b(MATRIX_SIZE * MATRIX_SIZE);
    vector<int> h_c(MATRIX_SIZE * MATRIX_SIZE);

    size_t bytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(int);

    CALI_MARK_BEGIN(data_init);
    generate(h_a.begin(), h_a.end(), []() { return rand() % 10 + 1; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 10 + 1; });
    CALI_MARK_END(data_init);

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    CALI_MARK_BEGIN(host2device);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
    CALI_MARK_END(host2device);

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);


    float start = clock();
    CALI_MARK_BEGIN(naive_time);
    naive<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CALI_MARK_END(naive_time);
    float stop = clock();
    print_elapsed(start,stop)

    CALI_MARK_BEGIN(device2host);
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    CALI_MARK_END(host2device);

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
    adiak::value("Algorithm", "CUDA Naive Matrix Multiplication");// The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA");          // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", int);                          // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));              // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", MATRIX_SIZE);                        // The number of elements in input dataset (1000)
    // adiak::value("InputType", inputType);                        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    //adiak::value("num_procs", size);                        // The number of processors (MPI ranks)
    adiak::value("num_threads", num_threads);                    // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", num_blocks);                      // The number of CUDA blocks
    adiak::value("group_num", 8);                     // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online");
    

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}