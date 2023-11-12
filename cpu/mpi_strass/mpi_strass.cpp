#include <mpi.h>
#include <bits/stdc++.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

const char *data_init = "data_init";
const char *comm = "comm";
const char *comm_small = "comm_small";
const char *comm_large = "comm_large";
const char *comp = "comp";
const char *comp_small = "comp_small";
const char *comp_large = "comp_large";
const char *correctness = "correctness";

const char *strassen_whole_computation = "strassen_whole_computation";
const char *bcast_n = "bcast_n";
const char *bcast_matricies = "bcast_matricies";
const char *splits = "splits";
const char *addsub = "addsub";
const char *combine = "combine";
const char *strassens = "strassens";
const char *worker_send = "worker_send";
const char *master_receive = "master_receive";
const char *mpi_barrier = "mpi_barrier";

void print(int n, int **mat)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf(" %d", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int **allocateMatrix(int n)
{
    int *data = (int *)malloc(n * n * sizeof(int));
    int **array = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++)
    {
        array[i] = &(data[n * i]);
    }
    return array;
}

void freeMatrix(int n, int **mat)
{
    free(mat[0]);
    free(mat);
}

int **naive(int n, int **mat1, int **mat2)
{
    int **prod = allocateMatrix(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            prod[i][j] = 0;
            for (int k = 0; k < n; k++)
            {
                prod[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return prod;
}

int **getSlice(int n, int **mat, int offseti, int offsetj)
{
    int m = n / 2;
    int **slice = allocateMatrix(m);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            slice[i][j] = mat[offseti + i][offsetj + j];
        }
    }
    return slice;
}

int **addMatrices(int n, int **mat1, int **mat2, bool add)
{
    int **result = allocateMatrix(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (add)
                result[i][j] = mat1[i][j] + mat2[i][j];
            else
                result[i][j] = mat1[i][j] - mat2[i][j];
        }
    }

    return result;
}

int **combineMatrices(int m, int **c11, int **c12, int **c21, int **c22)
{
    int n = 2 * m;
    int **result = allocateMatrix(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < m && j < m)
                result[i][j] = c11[i][j];
            else if (i < m)
                result[i][j] = c12[i][j - m];
            else if (j < m)
                result[i][j] = c21[i - m][j];
            else
                result[i][j] = c22[i - m][j - m];
        }
    }

    return result;
}

int **strassen(int n, int **mat1, int **mat2)
{

    if (n <= 32)
    {
        return naive(n, mat1, mat2);
    }

    int m = n / 2;

    int **a = getSlice(n, mat1, 0, 0); // A11
    int **b = getSlice(n, mat1, 0, m); // A12
    int **c = getSlice(n, mat1, m, 0); // A21
    int **d = getSlice(n, mat1, m, m); // A22
    int **e = getSlice(n, mat2, 0, 0); // B11
    int **f = getSlice(n, mat2, 0, m); // B12
    int **g = getSlice(n, mat2, m, 0); // B21
    int **h = getSlice(n, mat2, m, m); // B22

    int **fh_sub = addMatrices(m, f, h, false); // B12 - B22
    int **s1 = strassen(m, a, fh_sub);          // S1 = A11 * (B12 - B22)
    freeMatrix(m, fh_sub);

    int **ab_add = addMatrices(m, a, b, true); // A11 + A12
    int **s2 = strassen(m, ab_add, h);         // S2 = (A11 + A12) * B22
    freeMatrix(m, ab_add);

    int **cd_add = addMatrices(m, c, d, true); // A21 + A22
    int **s3 = strassen(m, cd_add, e);         // S3 = (A21 + A22) * B11
    freeMatrix(m, cd_add);

    int **ge_sub = addMatrices(m, g, e, false); // B21 - B11
    int **s4 = strassen(m, d, ge_sub);          // S4 = A22 * (B21 - B11)
    freeMatrix(m, ge_sub);

    int **ad_add = addMatrices(m, a, d, true); // A11 + A22
    int **eh_add = addMatrices(m, e, h, true); // B11 + B22
    int **s5 = strassen(m, ad_add, eh_add);    // S5 = (A11 + A22) * (B11 + B22)
    freeMatrix(m, ad_add);
    freeMatrix(m, eh_add);

    int **bd_sub = addMatrices(m, b, d, false); // B12 - B22
    int **gh_add = addMatrices(m, g, h, true);  // B21 + B22
    int **s6 = strassen(m, bd_sub, gh_add);     // S6 = (B12 - B22) * (B21 + B22)
    freeMatrix(m, bd_sub);
    freeMatrix(m, gh_add);

    int **ac_sub = addMatrices(m, a, c, false); // A11 - A21
    int **ef_add = addMatrices(m, e, f, true);  // B11 + B12
    int **s7 = strassen(m, ac_sub, ef_add);     // S7 = (A11 - A21) * (B11 + B12)
    freeMatrix(m, ac_sub);
    freeMatrix(m, ef_add);

    freeMatrix(m, a);
    freeMatrix(m, b);
    freeMatrix(m, c);
    freeMatrix(m, d);
    freeMatrix(m, e);
    freeMatrix(m, f);
    freeMatrix(m, g);
    freeMatrix(m, h);

    int **s4s5_add = addMatrices(m, s4, s5, true);         // S4 + S5
    int **s2s6_sub = addMatrices(m, s2, s6, false);        // S2 - S6
    int **c11 = addMatrices(m, s4s5_add, s2s6_sub, false); // C11 = S4 + S5 - S2 + S6
    freeMatrix(m, s4s5_add);
    freeMatrix(m, s2s6_sub);

    int **c12 = addMatrices(m, s1, s2, true); // S1 + S2

    int **c21 = addMatrices(m, s3, s4, true); // S3 + S4

    int **s1s5_add = addMatrices(m, s1, s5, true);         // S1 + S5
    int **s3s7_add = addMatrices(m, s3, s7, true);         // S3 + S7
    int **c22 = addMatrices(m, s1s5_add, s3s7_add, false); // C22 = S1 + S5 - S3 - S7
    freeMatrix(m, s1s5_add);
    freeMatrix(m, s3s7_add);

    freeMatrix(m, s1);
    freeMatrix(m, s2);
    freeMatrix(m, s3);
    freeMatrix(m, s4);
    freeMatrix(m, s5);
    freeMatrix(m, s6);
    freeMatrix(m, s7);

    int **prod = combineMatrices(m, c11, c12, c21, c22);

    freeMatrix(m, c11);
    freeMatrix(m, c12);
    freeMatrix(m, c21);
    freeMatrix(m, c22);

    return prod;
}

void strassen(int n, int **mat1, int **mat2, int **&prod, int rank)
{

    if (n == 1)
    {
        prod = allocateMatrix(1);
        prod[0][0] = mat1[0][0] * mat2[0][0];
    }

    int m = n / 2;
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(splits);
    int **a = getSlice(n, mat1, 0, 0); // A11
    int **b = getSlice(n, mat1, 0, m); // A12
    int **c = getSlice(n, mat1, m, 0); // A21
    int **d = getSlice(n, mat1, m, m); // A22
    int **e = getSlice(n, mat2, 0, 0); // B11
    int **f = getSlice(n, mat2, 0, m); // B12
    int **g = getSlice(n, mat2, m, 0); // B21
    int **h = getSlice(n, mat2, m, m); // B22
    CALI_MARK_END(splits);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    int **s1 = allocateMatrix(m);
    int **s2 = allocateMatrix(m);
    int **s3 = allocateMatrix(m);
    int **s4 = allocateMatrix(m);
    int **s5 = allocateMatrix(m);
    int **s6 = allocateMatrix(m);
    int **s7 = allocateMatrix(m);

    if (rank == 0)
    {
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);
        CALI_MARK_BEGIN(master_receive);
        MPI_Recv(&(s1[0][0]), m * m, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s2[0][0]), m * m, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s3[0][0]), m * m, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s4[0][0]), m * m, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s5[0][0]), m * m, MPI_INT, 5, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s6[0][0]), m * m, MPI_INT, 6, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s7[0][0]), m * m, MPI_INT, 7, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CALI_MARK_END(master_receive);
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);
    }

    if (rank == 1)
    {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        int **fh_sub = addMatrices(m, f, h, false); // B12 - B22
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(strassens);
        s1 = strassen(m, a, fh_sub); // S1 = A11 * (B12 - B22)
        CALI_MARK_END(strassens);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        freeMatrix(m, fh_sub);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        MPI_Send(&(s1[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }

    if (rank == 2)
    {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        int **ab_add = addMatrices(m, a, b, true); // A11 + A12
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(strassens);
        s2 = strassen(m, ab_add, h); // S2 = (A11 + A12) * B22
        CALI_MARK_END(strassens);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        freeMatrix(m, ab_add);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        MPI_Send(&(s2[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }

    if (rank == 3)
    {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        int **cd_add = addMatrices(m, c, d, true); // A21 + A22
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(strassens);
        s3 = strassen(m, cd_add, e); // S3 = (A21 + A22) * B11
        CALI_MARK_END(strassens);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        freeMatrix(m, cd_add);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        MPI_Send(&(s3[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }

    if (rank == 4)
    {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        int **ge_sub = addMatrices(m, g, e, false); // B21 - B11
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(strassens);
        s4 = strassen(m, d, ge_sub); // S4 = A22 * (B21 - B11)
        CALI_MARK_END(strassens);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        freeMatrix(m, ge_sub);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        MPI_Send(&(s4[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }

    if (rank == 5)
    {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        int **ad_add = addMatrices(m, a, d, true); // A11 + A22
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        int **eh_add = addMatrices(m, e, h, true); // B11 + B22
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(strassens);
        s5 = strassen(m, ad_add, eh_add); // S5 = (A11 + A22) * (B11 + B22)
        CALI_MARK_END(strassens);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        freeMatrix(m, ad_add);
        freeMatrix(m, eh_add);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        MPI_Send(&(s5[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }

    if (rank == 6)
    {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        int **bd_sub = addMatrices(m, b, d, false); // B12 - B22
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        int **gh_add = addMatrices(m, g, h, true); // B21 + B22
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(strassens);
        s6 = strassen(m, bd_sub, gh_add);
        CALI_MARK_END(strassens);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        freeMatrix(m, bd_sub);
        freeMatrix(m, gh_add);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        MPI_Send(&(s6[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }

    if (rank == 7)
    {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        int **ac_sub = addMatrices(m, a, c, false); // A11 - A21
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(addsub);
        int **ef_add = addMatrices(m, e, f, true); // B11 + B12
        CALI_MARK_END(addsub);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(strassens);
        s7 = strassen(m, ac_sub, ef_add);
        CALI_MARK_END(strassens);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        freeMatrix(m, ac_sub);
        freeMatrix(m, ef_add);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        MPI_Send(&(s7[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }

    freeMatrix(m, a);
    freeMatrix(m, b);
    freeMatrix(m, c);
    freeMatrix(m, d);
    freeMatrix(m, e);
    freeMatrix(m, f);
    freeMatrix(m, g);
    freeMatrix(m, h);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(mpi_barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(mpi_barrier);
    CALI_MARK_END(comm);

    if (rank == 0)
    {
        int **s4s5_add = addMatrices(m, s4, s5, true);         // S4 + S5
        int **s2s6_sub = addMatrices(m, s2, s6, false);        // S2 - S6
        int **c11 = addMatrices(m, s4s5_add, s2s6_sub, false); // C11 = S4 + S5 - S2 + S6
        freeMatrix(m, s4s5_add);
        freeMatrix(m, s2s6_sub);

        int **c12 = addMatrices(m, s1, s2, true); // S1 + S2

        int **c21 = addMatrices(m, s3, s4, true); // S3 + S4

        int **s1s5_add = addMatrices(m, s1, s5, true);         // S1 + S5
        int **s3s7_add = addMatrices(m, s3, s7, true);         // S3 + S7
        int **c22 = addMatrices(m, s1s5_add, s3s7_add, false); // C22 = S1 + S5 - S3 - S7
        freeMatrix(m, s1s5_add);
        freeMatrix(m, s3s7_add);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        CALI_MARK_BEGIN(combine);
        prod = combineMatrices(m, c11, c12, c21, c22);
        CALI_MARK_END(combine);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        freeMatrix(m, c11);
        freeMatrix(m, c12);
        freeMatrix(m, c21);
        freeMatrix(m, c22);
    }

    freeMatrix(m, s1);
    freeMatrix(m, s2);
    freeMatrix(m, s3);
    freeMatrix(m, s4);
    freeMatrix(m, s5);
    freeMatrix(m, s6);
    freeMatrix(m, s7);
}

int main(int argc, char *argv[])
{

    CALI_CXX_MARK_FUNCTION;

    int rank;
    int size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    if (argc == 2)
    {
        n = atoi(argv[1]);
    }
    else
    {
        printf("Please provide a matrix size\n");
        return 1;
    }
    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(mpi_barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(mpi_barrier);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(bcast_n);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(bcast_n);
    CALI_MARK_END(comm);

    int **mat1 = allocateMatrix(n);
    int **mat2 = allocateMatrix(n);

    if (rank == 0)
    {
        printf("Matrix Size: %d\n", n);
        printf("Number of Processes: %d\n", size);

        CALI_MARK_BEGIN(data_init);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                mat1[i][j] = rand() % 10 + 1;
                mat2[i][j] = rand() % 10 + 1;
            }
        }
        CALI_MARK_END(data_init);
        // print(n, mat1);
        // print(n, mat2);
    }

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(bcast_matricies);
    MPI_Bcast(&(mat1[0][0]), n * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(mat2[0][0]), n * n, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(bcast_matricies);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    double startTime = MPI_Wtime();

    int **prod;
    strassen(n, mat1, mat2, prod, rank);

    double endTime = MPI_Wtime();

    if (rank == 0)
    {
        printf("\nParallel Strassen Runtime (MPI): ");
        printf("%f\n\n", endTime - startTime);
        // print(n, prod);
        CALI_MARK_BEGIN(correctness);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int temp = 0;
                for (int k = 0; k < n; k++)
                {
                    temp += mat1[i][k] * mat2[k][j];
                }
                if (temp != prod[i][j])
                {
                    printf("Error at %d, %d\n", i, j);
                    printf("Expected: %d, Actual: %d\n", temp, prod[i][j]);
                    return 1;
                }
            }
        }
        CALI_MARK_END(correctness);
        printf("Verification Passed!\n");
    }

    adiak::init(NULL);
    adiak::launchdate();                                               // launch date of the job
    adiak::libraries();                                                // Libraries used
    adiak::cmdline();                                                  // Command line used to launch the job
    adiak::clustername();                                              // Name of the cluster
    adiak::value("Algorithm", "MPI Strassen's Matrix Multiplication"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");                           // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                                   // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));                       // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n);                                      // The number of elements in input dataset (1000)
    // adiak::value("InputType", inputType);                        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", size); // The number of processors (MPI ranks)
    // adiak::value("num_threads", num_threads);                    // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", num_blocks);                      // The number of CUDA blocks
    adiak::value("group_num", 8);                    // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    mgr.stop();
    mgr.flush();
    MPI_Finalize();

    return 0;
}