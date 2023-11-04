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
int 

int main(int argc, char *argv[]) {
    /*

    Assuming the vectors are created above
    */
    THREADS = atoi(argv[1]);
    MATRIX_SIZE = atoi(argv[2]);
    //Does it make sense to create a thread for every row/column?

    //So we know we have threads amount of threads so can we calculate how blocks we need then?

    BLOCKS = MATRIX_SIZE;
    //When we pass in the thread count it's talking about
    /*
    Issues that it's threads per block
    Essentially each block will take in threads amount of rows and cols and then send them to each block
    thus there will be (n*n)/threads blocks with thread threads inside of it
    */
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A,)
    return 0;
}