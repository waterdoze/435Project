#ifndef OMP_STRASS_H
#define OMP_STRASS_H

#include "omp_naive.h"
#include "../../common.h" //verify file path

#include "../../linear/lin_naive.h" // TODO: change to cuBlas when implemented

mat cpu_strass(mat &A, mat &B, int n) {
    if(n <= 32) {
        return cpu_naive(A, B, n);
    }

    int split_size = n / 2;

    mat a = split(n, A, 0, 0); // A11
    mat b = split(n, A, 0, split_size); // A12
    mat c = split(n, A, split_size, 0); // A21
    mat d = split(n, A, split_size, split_size); // A22

    mat e = split(n, B, 0, 0); // B11
    mat f = split(n, B, 0, split_size); // B12
    mat g = split(n, B, split_size, 0); // B21
    mat h = split(n, B, split_size, split_size); // B22

    mat s1;
    #pragma omp task shared(s1)
    {
        mat fh_sub = addsub_matricies(split_size, f, h, false);
        s1 = cpu_strass(a, fh_sub, split_size); // A11 * (B12 - B22)
    }

    mat s2;
    #pragma omp task shared(s2)
    {
        mat ab_add = addsub_matricies(split_size, a, b, true);
        s2 = cpu_strass(ab_add, h, split_size); // (A11 + A12) * B22
    }

    mat s3;
    #pragma omp task shared(s3)
    {
        mat cd_add = addsub_matricies(split_size, c, d, true);
        s3 = cpu_strass(cd_add, e, split_size); // (A21 + A22) * B11
    }

    mat s4;
    #pragma omp task shared(s4)
    {
        mat ge_sub = addsub_matricies(split_size, g, e, false);
        s4 = cpu_strass(d, ge_sub, split_size); // A22 * (B21 - B11)
    }

    mat s5;
    #pragma omp task shared(s5)
    {
        mat ad_add = addsub_matricies(split_size, a, d, true);
        mat eh_add = addsub_matricies(split_size, e, h, true);
        s5 = cpu_strass(ad_add, eh_add, split_size); // (A11 + A22) * (B11 + B22)
    }

    mat s6;
    #pragma omp task shared(s6)
    {
        mat bd_sub = addsub_matricies(split_size, b, d, false);
        mat gh_add = addsub_matricies(split_size, g, h, true);
        s6 = cpu_strass(bd_sub, gh_add, split_size); // (A12 - A22) * (B21 + B22)
    }

    mat s7;
    #pragma omp task shared(s7)
    {
        mat ac_sub = addsub_matricies(split_size, a, c, false);
        mat ef_add = addsub_matricies(split_size, e, f, true);
        s7 = cpu_strass(ac_sub, ef_add, split_size); // (A11 - A21) * (B11 + B12)
    }


    #pragma omp taskwait

    mat c11;
    #pragma omp task shared(c11)
    {
        mat s5_s4_add = addsub_matricies(split_size, s5, s4, true); // S5 + S4
        mat s2_s6_sub = addsub_matricies(split_size, s2, s6, false); // S2 - S6
        c11 = addsub_matricies(split_size, s5_s4_add, s2_s6_sub, false); // S5 + S4 - (S2 - S6) = S5 + S4 + S6 - S2
    }

    mat c12;
    #pragma omp task shared(c12)
    {
        c12 = addsub_matricies(split_size, s1, s2, true); // S1 + S2
    }

    mat c21;
    #pragma omp task shared(c21)
    {
        c21 = addsub_matricies(split_size, s3, s4, true); // S3 + S4
    }

    mat c22;
    #pragma omp task shared(c22)
    {
        mat s5_s1_add = addsub_matricies(split_size, s5, s1, true); // S5 + S1
        mat s3_s7_sub = addsub_matricies(split_size, s3, s7, true); // S3 + S7
        c22 = addsub_matricies(split_size, s5_s1_add, s3_s7_sub, false); // S5 + S1 - (S3 + S7) = S5 + S1 - S3 - S7
    }

    #pragma omp taskwait

    mat ret = combine_matricies(split_size, c11, c12, c21, c22);

    return ret;
}

#endif