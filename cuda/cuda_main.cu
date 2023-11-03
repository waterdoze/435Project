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

    BLOCKS = (MATRIX_SIZE*MATRIX_SIZE) / THREADS;
    flaot *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A,)
    return 0;
}