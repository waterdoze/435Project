#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#include "../../common.h"

#include "../../linear/lin_naive.h" // TODO: change to cuBlas when implemented
mat strassen(int n, mat m1, mat m2)
{
    // mat m3(n, std::vector<int>(n));
    // strassen(n, m1, m2, m3, 0);
    // return m3;

    if (n <= 32)
    {
        return cpu_naive(m1, m2, n);
    }

    int split_size = n / 2;

    mat a = split(n, m1, 0, 0);                   // A11
    mat b = split(n, m1, 0, split_size);          // A12
    mat c = split(n, m1, split_size, 0);          // A21
    mat d = split(n, m1, split_size, split_size); // A22

    mat e = split(n, m2, 0, 0);                   // B11
    mat f = split(n, m2, 0, split_size);          // B12
    mat g = split(n, m2, split_size, 0);          // B21
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

    mat s5_s4_add = addsub_matricies(split_size, s5, s4, true);            // S5 + S4
    mat s5_s4_s2_sub = addsub_matricies(split_size, s5_s4_add, s2, false); // S5 + S4 - S2
    mat c11 = addsub_matricies(split_size, s5_s4_s2_sub, s6, true);        // P1 = S5 + S4 - S2 + S6

    mat c12 = addsub_matricies(split_size, s1, s2, true); // S1 + S2

    mat c21 = addsub_matricies(split_size, s3, s4, true); // S3 + S4

    mat s5_s1_add = addsub_matricies(split_size, s5, s1, true);            // S5 + S1
    mat s5_s1_s3_sub = addsub_matricies(split_size, s5_s1_add, s3, false); // S5 + S1 - S3
    mat c22 = addsub_matricies(split_size, s5_s1_s3_sub, s7, false);       // P7 = S5 + S1 - S3 - S7

    return combine_matricies(split_size, c11, c12, c21, c22);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes

    int n;
    if (argc == 2) {
        n = atoi(argv[1]);
    } else {
        printf("needs size of matrix\n");
    }

    int local_n = n / size;

    mat a(n, std::vector<int>(n));
    mat b(n, std::vector<int>(n));
    mat c(n, std::vector<int>(n));

    if(rank == 0) {
        get_random_matrix(a);
        get_random_matrix(b);

        // scatter
        for(int i = 1; i < size; i++) {
            MPI_Send(&a[i * local_n][0], local_n * n, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&b[i * local_n][0], local_n * n, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&a[0][0], local_n * n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&b[0][0], local_n * n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    mat local_c(local_n, std::vector<int>(n));

    // do strassen
    local_c = strassen(local_n, a, b);

    // gather
    if(rank == 0) {
        for(int i = 0; i < local_n; i++) {
            for(int j = 0; j < n; j++) {
                c[i][j] = local_c[i][j];
            }
        }

        for(int i = 1; i < size; i++) {
            MPI_Recv(&c[i * local_n][0], local_n * n, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(&local_c[0][0], local_n * n, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}