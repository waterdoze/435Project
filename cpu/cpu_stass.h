#ifndef LIN_STRASS_H
#define LIN_STRASS_H

#include "cpu_naive.h"
#include "../common.h"



mat cpu_strass(mat &A, mat &B, int n) {
    if(n <= 32) {
        return cpu_naive(A, B, n);
    }

    int new_size = n / 2;

    mat a = split(n, A, 0, 0);
    mat b = split(n, A, 0, new_size);
    mat c = split(n, A, new_size, 0);
    mat d = split(n, A, new_size, new_size);

    mat e = split(n, B, 0, 0);
    mat f = split(n, B, 0, new_size);
    mat g = split(n, B, new_size, 0);
    mat h = split(n, B, new_size, new_size);

    mat s1;
    #pragma omp task shared(s1)
    {
        mat ad_add = addsub_matricies(new_size, a, d, true);
        mat eh_add = addsub_matricies(new_size, e, h, true);
        s1 = cpu_strass(ad_add, eh_add, new_size);
    }

    mat s2;
    #pragma omp task shared(s2)
    {
        mat bd_sub = addsub_matricies(new_size, b, d, false);
        mat gh_add = addsub_matricies(new_size, g, h, true);
        s2 = cpu_strass(bd_sub, gh_add, new_size);
    }

    mat s3;
    #pragma omp task shared(s3)
    {
        mat cd_add = addsub_matricies(new_size, c, d, true);
        s3 = cpu_strass(cd_add, e, new_size);
    }

    mat s4;
    #pragma omp task shared(s4)
    {
        mat ac_sub = addsub_matricies(new_size, a, c, false);
        mat ef_add = addsub_matricies(new_size, e, f, true);
        s4 = cpu_strass(ac_sub, ef_add, new_size);
    }

    mat s5;
    #pragma omp task shared(s5)
    {
        mat ge_sub = addsub_matricies(new_size, g, e, false);
        s5 = cpu_strass(d, ge_sub, new_size);
    }

    mat s6;
    #pragma omp task shared(s6)
    {
        mat ab_add = addsub_matricies(new_size, a, b, true);
        s6 = cpu_strass(ab_add, h, new_size);
    }

    mat s7;
    #pragma omp task shared(s7)
    {
        mat fh_sub = addsub_matricies(new_size, f, h, false);
        s7 = cpu_strass(a, fh_sub, new_size);
    }


    #pragma omp taskwait

    mat c11;
    #pragma omp task shared(c11)
    {
        mat s1_s2_add = addsub_matricies(new_size, s1, s2, true);
        mat s6_s4_sub = addsub_matricies(new_size, s6, s4, false);
        c11 = addsub_matricies(new_size, s1_s2_add, s6_s4_sub, true);
    }

    mat c12;
    #pragma omp task shared(c12)
    {
        c12 = addsub_matricies(new_size, s4, s5, true);
    }

    mat c21;
    #pragma omp task shared(c21)
    {
        c21 = addsub_matricies(new_size, s6, s7, true);
    }

    mat c22;
    #pragma omp task shared(c22)
    {
        mat s2_s3_sub = addsub_matricies(new_size, s2, s3, false);
        mat s5_s7_sub = addsub_matricies(new_size, s5, s7, false);
        c22= addsub_matricies(new_size, s2_s3_sub, s5_s7_sub, true);
    }

    #pragma omp taskwait

    mat ret = combine_matricies(n, c11, c12, c21, c22);

    return ret;
}

#endif