#ifndef LIN_STRASS_H
#define LIN_STRASS_H

#include "cpu_naive.h"
#include "../common.h"



mat cpu_strass(mat &A, mat &B, int n) {
    if(n <= 32) {
        return cpu_naive(A, B, n);
    }

    int split_size = n / 2;

    mat a = split(n, A, 0, 0);
    mat b = split(n, A, 0, split_size);
    mat c = split(n, A, split_size, 0);
    mat d = split(n, A, split_size, split_size);

    mat e = split(n, B, 0, 0);
    mat f = split(n, B, 0, split_size);
    mat g = split(n, B, split_size, 0);
    mat h = split(n, B, split_size, split_size);

    mat s1;
    #pragma omp task shared(s1)
    {
        mat fh_sub = addsub_matricies(split_size, f, h, false);
        s1 = cpu_strass(a, fh_sub, split_size);
    }

    mat s2;
    #pragma omp task shared(s2)
    {
        mat ab_add = addsub_matricies(split_size, a, b, true);
        s2 = cpu_strass(ab_add, h, split_size);
    }

    mat s3;
    #pragma omp task shared(s3)
    {
        mat cd_add = addsub_matricies(split_size, c, d, true);
        s3 = cpu_strass(cd_add, e, split_size);
    }

    mat s4;
    #pragma omp task shared(s4)
    {
        mat ge_sub = addsub_matricies(split_size, g, e, false);
        s4 = cpu_strass(d, ge_sub, split_size);
    }

    mat s5;
    #pragma omp task shared(s5)
    {
        mat ad_add = addsub_matricies(split_size, a, d, true);
        mat eh_add = addsub_matricies(split_size, e, h, true);
        s5 = cpu_strass(ad_add, eh_add, split_size);

    }

    mat s6;
    #pragma omp task shared(s6)
    {
        mat bd_sub = addsub_matricies(split_size, b, d, false);
        mat gh_add = addsub_matricies(split_size, g, h, true);
        s6 = cpu_strass(bd_sub, gh_add, split_size);

    }

    mat s7;
    #pragma omp task shared(s7)
    {
        mat ac_sub = addsub_matricies(split_size, a, c, false);
        mat ef_add = addsub_matricies(split_size, e, f, true);
        s7 = cpu_strass(ac_sub, ef_add, split_size);
    }


    #pragma omp taskwait

    mat c11;
    #pragma omp task shared(c11)
    {
        mat s5_s4_add = addsub_matricies(split_size, s5, s4, true);
        mat s2_s6_add = addsub_matricies(split_size, s2, s6, true);
        c11 = addsub_matricies(split_size, s5_s4_add, s2_s6_add, false);
    }

    mat c12;
    #pragma omp task shared(c12)
    {
        c12 = addsub_matricies(split_size, s1, s2, true);
    }

    mat c21;
    #pragma omp task shared(c21)
    {
        c21 = addsub_matricies(split_size, s3, s4, true);
    }

    mat c22;
    #pragma omp task shared(c22)
    {
        mat s1_s5_add = addsub_matricies(split_size, s1, s5, true);
        mat s3_s7_sub = addsub_matricies(split_size, s3, s7, false);
        c22 = addsub_matricies(split_size, s1_s5_add, s3_s7_sub, true);
    }

    #pragma omp taskwait

    mat ret = combine_matricies(split_size, c11, c12, c21, c22);

    return ret;
}

#endif