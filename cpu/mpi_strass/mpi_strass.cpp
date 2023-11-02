#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#include "../../common.h"

#include "../../linear/lin_naive.h" // TODO: change to cuBlas when implemented

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

mat strassen(int n, mat m1, mat m2);
void strassen(int n, mat m1, mat m2, mat &m3, int taskid);


int main(int argc, char* argv[]) {
    int taskid, numtasks;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    int n;
    if (argc != 2) {
        std::cout << "include matrix size" << std::endl;
        return 1;
    }
    n = atoi(argv[1]);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    mat a, b, c;

    if(taskid == MASTER) {
        get_random_matrix(a);
        get_random_matrix(b);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&a[0][0], n*n, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&b[0][0], n*n, MPI_INT, MASTER, MPI_COMM_WORLD);

    strassen(n, a, b, c, taskid);

    MPI_Finalize();

}

mat strassen(int n, mat m1, mat m2) {
    // mat m3(n, std::vector<int>(n));
    // strassen(n, m1, m2, m3, 0);
    // return m3;

    if(n <= 32) {
        return cpu_naive(m1, m2, n);
    }

    int split_size = n / 2;

    mat a = split(n, m1, 0, 0); // A11
    mat b = split(n, m1, 0, split_size); // A12
    mat c = split(n, m1, split_size, 0); // A21
    mat d = split(n, m1, split_size, split_size); // A22

    mat e = split(n, m2, 0, 0); // B11
    mat f = split(n, m2, 0, split_size); // B12
    mat g = split(n, m2, split_size, 0); // B21
    mat h = split(n, m2, split_size, split_size); // B22

    mat fh_sub = addsub_matricies(split_size, f, h, false);
    mat s1 = strassen(split_size, a, fh_sub); // A11 * (B12 - B22)

    mat ab_add = addsub_matricies(split_size, a, b, true);
    mat s2 = strassen(split_size, ab_add, h); // (A11 + A12) * B22

    mat cd_add = addsub_matricies(split_size, c, d, true);
    mat s3 = strassen(split_size, cd_add, e); // (A21 + A22) * B11

    mat ge_sub = addsub_matricies(split_size, g, e, false);
    mat s4 = strassen(split_size, d, ge_sub); // A22 * (B21 - B11)

    mat ad_add = addsub_matricies(split_size, a, d, true);
    mat eh_add = addsub_matricies(split_size, e, h, true);
    mat s5 = strassen(split_size, ad_add, eh_add); // (A11 + A22) * (B11 + B22)

    mat bd_sub = addsub_matricies(split_size, b, d, false);
    mat gh_add = addsub_matricies(split_size, g, h, true);
    mat s6 = strassen(split_size, bd_sub, gh_add); // (A12 - A22) * (B21 + B22)

    mat ac_sub = addsub_matricies(split_size, a, c, false);
    mat ef_add = addsub_matricies(split_size, e, f, true);
    mat s7 = strassen(split_size, ac_sub, ef_add); // (A11 - A21) * (B11 + B12)

    mat s5_s4_add = addsub_matricies(split_size, s5, s4, true); // S5 + S4
    mat s5_s4_s2_sub = addsub_matricies(split_size, s5_s4_add, s2, false); // S5 + S4 - S2
    mat c11 = addsub_matricies(split_size, s5_s4_s2_sub, s6, true); // P1 = S5 + S4 - S2 + S6

    mat c12 = addsub_matricies(split_size, s1, s2, true); // S1 + S2

    mat c21 = addsub_matricies(split_size, s3, s4, true); // S3 + S4

    mat s5_s1_add = addsub_matricies(split_size, s5, s1, true); // S5 + S1
    mat s5_s1_s3_sub = addsub_matricies(split_size, s5_s1_add, s3, false); // S5 + S1 - S3
    mat c22 = addsub_matricies(split_size, s5_s1_s3_sub, s7, false); // P7 = S5 + S1 - S3 - S7

    return combine_matricies(split_size, c11, c12, c21, c22);
}

void strassen(int n, mat m1, mat m2, mat& m3, int taskid) {
    if (n <= 32) {
        return cpu_naive(A, B, n); // TODO: change to cuBlas when implemented
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

    mat s1, s2, s3, s4, s5, s6, s7;

    if(taskid == 0) {
        MPI_Recv(&(s1[0][0]), split_size*split_size, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s2[0][0]), split_size*split_size, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s3[0][0]), split_size*split_size, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s4[0][0]), split_size*split_size, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s5[0][0]), split_size*split_size, MPI_INT, 5, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s6[0][0]), split_size*split_size, MPI_INT, 6, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s7[0][0]), split_size*split_size, MPI_INT, 7, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if(taskid == 1) {
        mat fh_sub = addsub_matricies(split_size, f, h, false);
        s1 = strassen(split_size, a, fh_sub);
        MPI_Send(&(s1[0][0]), split_size*split_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    if(taskid == 2) {
        mat ab_add = addsub_matricies(split_size, a, b, true);
        s2 = strassen(split_size, ab_add, h);
        MPI_Send(&(s2[0][0]), split_size*split_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    if(taskid == 3) {
        mat cd_add = addsub_matricies(split_size, c, d, true);
        s3 = strassen(split_size, cd_add, e);
        MPI_Send(&(s3[0][0]), split_size*split_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    if(taskid == 4) {
        mat ge_sub = addsub_matricies(split_size, g, e, false);
        s4 = strassen(split_size, d, ge_sub);
        MPI_Send(&(s4[0][0]), split_size*split_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    if(taskid == 5) {
        mat ad_add = addsub_matricies(split_size, a, d, true);
        mat eh_add = addsub_matricies(split_size, e, h, true);
        s5 = strassen(split_size, ad_add, eh_add);
        MPI_Send(&(s5[0][0]), split_size*split_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    if(taskid == 6) {
        mat bd_sub = addsub_matricies(split_size, b, d, false);
        mat eg_add = addsub_matricies(split_size, e, g, true);
        s6 = strassen(split_size, bd_sub, eg_add);
        MPI_Send(&(s6[0][0]), split_size*split_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    if(taskid == 7) {
        mat ac_sub = addsub_matricies(split_size, a, c, false);
        mat eg_add = addsub_matricies(split_size, e, g, true);
        s7 = strassen(split_size, ac_sub, eg_add);
        MPI_Send(&(s7[0][0]), split_size*split_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(taskid == 0) {
        mat s5_s4_add = addsub_matricies(split_size, s5, s4, true); // S5 + S4
        mat s5_s4_s2_sub = addsub_matricies(split_size, s5_s4_add, s2, false); // S5 + S4 - S2
        mat c11 = addsub_matricies(split_size, s5_s4_s2_sub, s6, true); // P1 = S5 + S4 - S2 + S6

        mat c12 = addsub_matricies(split_size, s1, s2, true); // S1 + S2

        mat c21 = addsub_matricies(split_size, s3, s4, true); // S3 + S4

        mat s5_s1_add = addsub_matricies(split_size, s5, s1, true); // S5 + S1
        mat s5_s1_s3_sub = addsub_matricies(split_size, s5_s1_add, s3, false); // S5 + S1 - S3
        mat c22 = addsub_matricies(split_size, s5_s1_s3_sub, s7, false); // P7 = S5 + S1 - S3 - S7

        m3 = combine_matricies(split_size, c11, c12, c21, c22);
    }
}