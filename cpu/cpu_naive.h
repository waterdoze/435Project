#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef CPU_NAIVE_H
#define CPU_NAIVE_H

def cpu_naive(std::vector<std::vector<float>> &A, std::vector<std::vector<float>> &B, std::vector<std::vector<float>> &C) {
    size_t A_row = A.size();
    size_t A_column = A[0].size();
    size_t B_row = B.size();
    size_t B_column = B[0].size();
    size_t C_row = C.size();
    size_t C_column = C[0].size();

    #pragma omp parallel for private(i, j, k) shared(A, B, C)
    for (size_t i = 0; i < A_row; ++i) {
        for (size_t j = 0; j < B_column; ++j) {
            for (size_t k = 0; k < A_column; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}