#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <ctime>

#include "../../common.h" //verify file path
#include "omp_naive.h"
#include "omp_strass.h"

int main(int argc, char *argv[]){
    if (argc != 2) {
        std::cout << "Usage: ./cpu_main <matrix_size> <num_processes>" << std::endl;
        return 1;
    }

    int mat_size = atoi(argv[1]);
    int num_processes = atoi(argv[2]);

    size_t A_row = mat_size, A_column = mat_size, B_row = mat_size,
           B_column = mat_size;

    std::vector<std::vector<int>> A(A_row, std::vector<int>(A_column));
    std::vector<std::vector<int>> B(B_row, std::vector<int>(B_column));

    get_random_matrix(A);

    get_random_matrix(B);

    std::vector<std::vector<int>> C(A_row, std::vector<int>(B_column));

    omp_set_num_threads(num_processes);
    std::clock_t start = std::clock();
    cpu_naive(A, B, C);
    double duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    printf("cpu naive time span: %f\n", duration);

    return 0;
}