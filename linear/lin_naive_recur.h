#ifndef LIN_NAIVE_RECUR_H
#define LIN_NAIVE_RECUR_H

void cpu_lin_naive_recur(const mat &A, const mat &B,
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

    mat part_one(dim, std::vector<int>(dim)), part_two(dim, std::vector<int>(dim));


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


    // divide original matrix into 4 sub-matrix
    // C11 = A11 * B11 + A12 * B21
    // C12 = A11 * B12 + A12 * B22
    // C21 = A21 * B11 + A22 * B21
    // C22 = A21 * B12 + A22 * B22


    cpu_lin_naive_recur(A11, B11, part_one);
    cpu_lin_naive_recur(A12, B21, part_two);
    matrix_add(part_one, part_two, C11);

    cpu_lin_naive_recur(A11, B12, part_one);
    cpu_lin_naive_recur(A12, B22, part_two);
    matrix_add(part_one, part_two, C12);

    cpu_lin_naive_recur(A21, B11, part_one);
    cpu_lin_naive_recur(A22, B21, part_two);
    matrix_add(part_one, part_two, C21);

    cpu_lin_naive_recur(A21, B12, part_one);
    cpu_lin_naive_recur(A22, B22, part_two);
    matrix_add(part_one, part_two, C22);



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