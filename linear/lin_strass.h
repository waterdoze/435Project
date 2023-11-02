#ifndef LIN_STRASS_H
#define LIN_STRASS_H
#include "../common.h"

void cpu_strass_naive(mat &A, mat &B, mat &C)
{
    // Assume N X N Matrix and 2^Z = N
    size_t N = A.size();
    if (N != A.at(0).size() || N != B.size() || N != B.at(0).size()) {
        std::cout << "ERROR: Matrix is not square" << std::endl;
        return;
    }

    // Base Case

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
    mat p1 = cpu_strass_naive(a, addsub_matricies(N / 2, f, h, false));
    /*
    p1 = a(f-h)
    p2 = (a+b)h
    p3 = (c+d)e
    p4 = d(g-e)
    p5 = (a+d)(e+h)
    p6 = (b-d)(g+h)
    p7 = (a-c)(e+f)
    */

   // Combine to Form C Quadrants

   // Construct C and Return
}
#endif