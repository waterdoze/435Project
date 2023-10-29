#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <ctime>
#include "common.h"
#include "./linear/lin_naive.h"

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


int main(int argc, char *argv[]) {
	
    int mat_size = atoi(argv[1]);

    size_t A_row = mat_size, A_column = mat_size, B_row = mat_size,
           B_column = mat_size;
    std::vector<std::vector<float>> A(A_row, std::vector<float>(A_column));
    std::vector<std::vector<float>> B(B_row, std::vector<float>(B_column));

    float min = 1.0, max = 100.0;
    get_random_matrix(A, min, max);
    // print_matrix<float>(A, "A");

    get_random_matrix(B, min, max);
    // print_matrix<float>(B, "B");

    std::vector<std::vector<float>> C(A_row, std::vector<float>(B_column));

    std::clock_t start = std::clock();
    matmul_naive_cpu(A, B, C);
    double duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    std::cout << "time span: " << duration << std::endl;
}


