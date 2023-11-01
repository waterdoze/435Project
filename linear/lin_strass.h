#ifndef LIN_NAIVE_H
#define LIN_NAIVE_H
#include "../common.h"

void cpu_lin_strass(mat &A, mat &B, mat &C) {
    size_t A_row = A.size();
	size_t A_column = A[0].size();
	size_t B_row = B.size();
	size_t B_column = B[0].size();
	size_t C_row = C.size();
	size_t C_column = C[0].size();

    if (A_row == 1) {
        C[0][0] = A[0][0] * B[0][0];
        return;
    }

    size_t dim = A_row / 2;

    mat A11(dim, std::vector<int>(dim)), 
        A12(dim, std::vector<int>(dim)),
        A21(dim, std::vector<int>(dim)),
        A22(dim, std::vector<int>(dim));

    mat B11(dim, std::vector<int>(dim)), 
        B12(dim, std::vector<int>(dim)),
        B21(dim, std::vector<int>(dim)),
        B22(dim, std::vector<int>(dim));  

    mat C11(dim, std::vector<int>(dim)), 
        C12(dim, std::vector<int>(dim)),
        C21(dim, std::vector<int>(dim)),
        C22(dim, std::vector<int>(dim));

    mat P1(dim, std::vector<int>(dim)), 
        P2(dim, std::vector<int>(dim)),
        P3(dim, std::vector<int>(dim)),
        P4(dim, std::vector<int>(dim)),
        P5(dim, std::vector<int>(dim)),
        P6(dim, std::vector<int>(dim)),
        P7(dim, std::vector<int>(dim));

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + dim];
            A21[i][j] = A[i + dim][j];
            A22[i][j] = A[i + dim][j + dim];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + dim];
            B21[i][j] = B[i + dim][j];
            B22[i][j] = B[i + dim][j + dim];
        }
    }
}

#endif