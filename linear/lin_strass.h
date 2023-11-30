#ifndef LIN_STRASS_H
#define LIN_STRASS_H

void matrix_add(const mat &A, const mat &B,
                mat &result) {
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
}

void matrix_sub(const mat &A, const mat &B,
                mat &result) {
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
}

void matrix_multiply_strassen(const mat &A, const mat &B,
                              mat &C) {
    size_t A_row = A.size();
    size_t A_column = A[0].size();
    size_t B_row = B.size();
    size_t B_column = B[0].size();
    size_t C_row = C.size();
    size_t C_column = C[0].size();
    if (A_column != B_row || A_row != C_row || B_column != C_column) {
        printf("input error: A (%zu * %zu) * B (%zu * %zu) != C (%zu * %zu)", A_row, A_column, B_row, B_column, C_row,
             C_column);
        return;
    }

    if (A_row <= 512) {
        cpu_lin_naive(A, B, C);
        return;
    }

    size_t dim = A_row / 2;

    mat A11(dim, std::vector<int>(dim)), A12(dim, std::vector<int>(dim)),
        A21(dim, std::vector<int>(dim)), A22(dim, std::vector<int>(dim));
    mat B11(dim, std::vector<int>(dim)), B12(dim, std::vector<int>(dim)),
        B21(dim, std::vector<int>(dim)), B22(dim, std::vector<int>(dim));
    mat C11(dim, std::vector<int>(dim)), C12(dim, std::vector<int>(dim)),
        C21(dim, std::vector<int>(dim)), C22(dim, std::vector<int>(dim));
    mat P1(dim, std::vector<int>(dim)), P2(dim, std::vector<int>(dim)),
        P3(dim, std::vector<int>(dim)), P4(dim, std::vector<int>(dim)), P5(dim, std::vector<int>(dim)),
        P6(dim, std::vector<int>(dim)), P7(dim, std::vector<int>(dim));
    mat A_result(dim, std::vector<int>(dim)), B_result(dim, std::vector<int>(dim));

    // divide original matrix into 4 sub-matrix
    // C11 = A11 * B11 + A12 * B21
    // C12 = A11 * B12 + A12 * B22
    // C21 = A21 * B11 + A22 * B21
    // C22 = A21 * B12 + A22 * B22
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
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

    // Calculate P1 to P7
    matrix_add(A11, A22, A_result);                    // A11 + A22
    matrix_add(B11, B22, B_result);                    // B11 + B22
    matrix_multiply_strassen(A_result, B_result, P1);  // P1 = (A11 + A22) * (B11 + B22)

    matrix_add(A21, A22, A_result);               // A21 + A22
    matrix_multiply_strassen(A_result, B11, P2);  // P2 = (A21 + A22) * B11

    matrix_sub(B12, B22, B_result);               // B12 - B22
    matrix_multiply_strassen(A11, B_result, P3);  // P3 = A11 * (B12 - B22)

    matrix_sub(B21, B11, B_result);               // B21 - B11
    matrix_multiply_strassen(A22, B_result, P4);  // P4 = A22 * (B21 - B11)

    matrix_add(A11, A12, A_result);               // A11 + A12
    matrix_multiply_strassen(A_result, B22, P5);  // P5 = (A11 + A12) * B22

    matrix_sub(A21, A11, A_result);                    // A21 - A11
    matrix_add(B11, B12, B_result);                    // B11 + B12
    matrix_multiply_strassen(A_result, B_result, P6);  // P6 = (A21 - A11) * (B11 + B12)

    matrix_sub(A12, A22, A_result);                    // A12 - A22
    matrix_add(B21, B22, B_result);                    // B21 + B22
    matrix_multiply_strassen(A_result, B_result, P7);  // p7 = (A12 - A22) * (B21 + B22)

    // calculate C11, C12, C21 and C22:
    matrix_add(P1, P4, A_result);        // P1 + P4
    matrix_add(A_result, P7, B_result);  // P1 + P4 + P7
    matrix_sub(B_result, P5, C11);       // C11 = P1 + P4 - P5 + P7

    matrix_add(P3, P5, C12);  // C12 = P3 + P5

    matrix_add(P2, P4, C21);  // C21 = P2 + P4

    matrix_add(P1, P3, A_result);        // P1 + P3
    matrix_add(A_result, P6, B_result);  // P1 + P3 + P6
    matrix_sub(B_result, P2, C22);       // C22 = P1 + P3 - P2 + P6

    // put results together
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + dim] = C12[i][j];
            C[i + dim][j] = C21[i][j];
            C[i + dim][j + dim] = C22[i][j];
        }
    }
}
#endif