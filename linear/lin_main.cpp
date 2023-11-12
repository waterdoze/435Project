#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <ctime>
#include "../common.h"
#include "lin_naive.h"
#include "lin_strass.h"
#include "lin_naive_recur.h"

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

bool assertMatrixEqual(const mat &m, const mat &);

int main(int argc, char *argv[]) {
	
    CALI_CXX_MARK_FUNCTION;

    int mat_size = atoi(argv[1]);

    size_t A_row = mat_size, A_column = mat_size, B_row = mat_size,
           B_column = mat_size;
    mat A(A_row, std::vector<int>(A_column));
    mat B(B_row, std::vector<int>(B_column));

    CALI_MARK_BEGIN("data_init");

    get_random_matrix(A);
    // print_matrix(A);

    get_random_matrix(B);
    // print_matrix(B);


    mat C(A_row, std::vector<int>(B_column));

    CALI_MARK_END("data_init");

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    std::clock_t start = std::clock();
    cpu_lin_naive_recur(A, B, C);
    double duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");
    std::cout << "linear naive time span: " << duration << std::endl;

    CALI_MARK_BEGIN("correctness_check");
    mat C_compare(A_row, std::vector<int>(B_column));
    cpu_lin_naive(A, B, C_compare);

    std::cout << "algorithm has " << (assertMatrixEqual(C, C_compare) ? "correctly" : "incorrectly") 
                                  << " calculated the resulting matrix" << std::endl;

    CALI_MARK_END("correctness_check");
    
    adiak::init(NULL);
    adiak::launchdate();                                         // launch date of the job
    adiak::libraries();                                          // Libraries used
    adiak::cmdline();                                            // Command line used to launch the job
    adiak::clustername();                                        // Name of the cluster
    adiak::value("Algorithm", "Linear Naive with Recursion");// The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "C++");          // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                          // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));              // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", mat_size);                        // The number of elements in input dataset (1000)
    // adiak::value("InputType", inputType);                        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", 1);                        // The number of processors (MPI ranks)
    // adiak::value("num_threads", num_threads);                    // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", num_blocks);                      // The number of CUDA blocks
    adiak::value("group_num", 8);                     // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
}

bool assertMatrixEqual(const mat &mat1, const mat &mat2) {
    if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size()) {
        return false;
    }

    for (int i = 0; i < mat1.size(); i++) {
        for (int j = 0; j < mat2.size(); j++) {
            if (mat1[i][j] != mat2[i][j]) {
                return false;
            }
        }
    }

    return true;
}
