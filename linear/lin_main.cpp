#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <ctime>
#include "../common.h"
#include "./lin_naive.h"

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


int main(int argc, char *argv[]) {
	
    int mat_size = atoi(argv[1]);

    size_t A_row = mat_size, A_column = mat_size, B_row = mat_size,
           B_column = mat_size;
    std::vector<std::vector<int>> A(A_row, std::vector<int>(A_column));
    std::vector<std::vector<int>> B(B_row, std::vector<int>(B_column));

    get_random_matrix(A);
    print_matrix(A);

    get_random_matrix(B);
    print_matrix(B);

    std::vector<std::vector<int>> C(A_row, std::vector<int>(B_column));

    std::clock_t start = std::clock();
    cpu_lin_naive(A, B, C);
    double duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    std::cout << "linear naive time span: " << duration << std::endl;
}


