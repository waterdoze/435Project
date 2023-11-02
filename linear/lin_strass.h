#ifndef LIN_STRASS_H
#define LIN_STRASS_H
#include "../common.h"

void cpu_strass_naive(mat &A, mat &B, mat &C) {
    size_t N = A.size();
    if (N != A.at(0).size() || N != B.size() || N != B.at(0).size()) {
        std::cout << "ERROR: Matrix is not square" << std::endl;
        return;
    }

    if (N <= 1) {
        std::cout << "ERROR: Matrix size is less than or equal to 1" << std::endl;
        return;
    }

    if (N & (N - 1)) {
        std::cout << "ERROR: Matrix size is not a power of 2" << std::endl;
        return;
    }

    C = strassen(A, B);
}

mat strassen(mat A, mat B) {

    // Base Case
    if (N == 2) {
        mat C(2, std::vector<int>(2));
        C[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0];
        C[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1];
        C[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0];
        C[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1];
        return;
    }

    // Split A & B
    mat a = split(N, A, 0, 0);
    mat b = split(N, A, 0, N / 2);
    mat c = split(N, A, N / 2, 0);
    mat d = split(N, A, N / 2, N / 2);

    mat e = split(N, B, 0, 0);
    mat f = split(N, B, 0, N / 2);
    mat g = split(N, B, N / 2, 0);
    mat h = split(N, B, N / 2, N / 2);

    // Calculate P1 - P7
    mat p1 = strassen(a, subM(f, h)); // a(f - h)
    mat p2 = strassen(addM(a, b), h); // (a + b)h
    mat p3 = strassen(addM(c, d), e); // (c + d)e
    mat p4 = strassen(d, subM(g, e)); // d(g - e)
    mat p5 = strassen(addM(a, d), addM(e, h)); // (a + d)(e + h)
    mat p6 = strassen(subM(b, d), addM(g, h)); // (b - d)(g + h)
    mat p7 = strassen(subM(a, c), addM(e, f)); // (a - c)(e + f)

    // Combine to Form C Quadrants
    mat q1 = addM(subM(addM(p5, p4), p2), p6); // p5 + p4 - p2 + p6
    mat q2 = addM(p1, p2); // p1 + p2
    mat q3 = addM(p3, p4); // p3 + p4
    mat q4 = subM(subM(addM(p1, p5), p3), p7); // p1 + p5 - p3 - p7

    // Construct C and Return
    mat C = combine_matricies(q1.size(), q1, q2, q3, q4);
    return C;
}
#endif