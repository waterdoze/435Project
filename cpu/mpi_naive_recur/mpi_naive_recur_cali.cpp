#include <mpi.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <iostream>

#define IDX(i, j, n) ((i) * (n) + (j))

const char* data_init = "data_init";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* correctness = "correctness";

const char* add_matrix = "add_matrix";
const char* copy_quadrant = "copy_quadrant";
const char* master_send = "master_send";
const char* master_recv = "master_recv";
const char* worker_send = "worker_send";
const char* naive_recursion = "naive_recursion";
const char* combine = "combine";


void batchFreeM(int *arrays[], int num_arrays)
{
    for (int i = 0; i < num_arrays; ++i)
        delete[] arrays[i];
    delete[] matrix;
}

void copyQuadrant(int src_n, int *src, int *dest, int quadrant)
{
    // quad =
    // 1 | 2
    // 3 | 4
    int dest_n = src_n / 2;
    int row_offset = (quadrant == 1 || quadrant == 2) ? 0 : dest_n;
    int col_offset = (quadrant == 1 || quadrant == 3) ? 0 : dest_n;
    for (int i = 0; i < dest_n; ++i)
        for (int j = 0; j < dest_n; ++j)
            dest[IDX(i, j, dest_n)] = src[IDX(i + row_offset, j + col_offset, src_n)];
}

void addM(int n, int *a, int *b)
{
    for (int i = 0; i < n * n; ++i)
        a[i] += b[i];
}

int *naive_recursive_mult(int N, int *A, int *B)
{
    int quadN = N / 2;

    // allocate C
    int **C = allocateM(N);

    // base case
    if (N == 1)
    {
        C[0][0] = A[0][0] * B[0][0];
        return C;
    }

    // allocate & init 8 quads
    int *A1 = new int[quadN * quadN];
    int *A2 = new int[quadN * quadN];
    int *A3 = new int[quadN * quadN];
    int *A4 = new int[quadN * quadN];
    int *B1 = new int[quadN * quadN];
    int *B2 = new int[quadN * quadN];
    int *B3 = new int[quadN * quadN];
    int *B4 = new int[quadN * quadN];

    copyQuadrant(N, A, A1, 1);
    copyQuadrant(N, A, A2, 2);
    copyQuadrant(N, A, A3, 3);
    copyQuadrant(N, A, A4, 4);
    copyQuadrant(N, B, B1, 1);
    copyQuadrant(N, B, B2, 2);
    copyQuadrant(N, B, B3, 3);
    copyQuadrant(N, B, B4, 4);

    // multiply quads
    int *C1_part1 = naive_recursive_mult(quadN, A1, B1);
    int *C1_part2 = naive_recursive_mult(quadN, A2, B3);
    int *C2_part1 = naive_recursive_mult(quadN, A1, B2);
    int *C2_part2 = naive_recursive_mult(quadN, A2, B4);
    int *C3_part1 = naive_recursive_mult(quadN, A3, B1);
    int *C3_part2 = naive_recursive_mult(quadN, A4, B3);
    int *C4_part1 = naive_recursive_mult(quadN, A3, B2);
    int *C4_part2 = naive_recursive_mult(quadN, A4, B4);

    // add quad pairs
    addM(quadN, C1_part1, C1_part2);
    addM(quadN, C2_part1, C2_part2);
    addM(quadN, C3_part1, C3_part2);
    addM(quadN, C4_part1, C4_part2);

    // combine to C
    for (int i = 0; i < quadN; ++i)
        for (int j = 0; j < quadN; ++j)
        {
            C[IDX(i, j, N)] = C1_part1[IDX(i, j, quadN)];
            C[IDX(i, j + quadN, N)] = C2_part1[IDX(i, j, quadN)];
            C[IDX(i + quadN, j, N)] = C3_part1[IDX(i, j, quadN)];
            C[IDX(i + quadN, j + quadN, N)] = C4_part1[IDX(i, j, quadN)];
        }

    // free allocated A & B quads & returned C parts
    // ! Could fail idk, just check if compilation error happens
    // TODO: if fails, just do delete[] A1; delete[] A2; ... delete[] C4_part2;
    batchFreeM(
        new int *[]
        {
            A1, A2, A3, A4, B1, B2, B3, B4,
                C1_part1, C1_part2, C2_part1, C2_part2, C3_part1, C3_part2, C4_part1, C4_part2
        },
        16);

    // return C
    return C;
}

int main(int argc, char *argv[])
{

    CALI_CXX_MARK_FUNCTION;

    MPI_Init(&argc, &argv);

    int processors;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n; // matrix size
    if (argc == 2)
    {
        n = atoi(argv[1]);
    }
    else
    {
        printf("Please provide a matrix size\n");
        return 1;
    }

    // Matrix Problem (unused by ranks 1-7)
    int *A;
    int *B;
    int *C; // result matrix C

    if (rank == 0)
    {
        // initialize matrices
        A = new int[n * n];
        B = new int[n * n];
        C = new int[n * n];
        CALI_MARK_BEGIN(data_init);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; ++j)
            {
                A[i][j] = rand() % 10 + 1;
                B[i][j] = rand() % 10 + 1;
            }
        }
        CALI_MARK_END(data_init);

        // split into 8 pieces
        int **A1 = new int[n * n / 4];
        int **A2 = new int[n * n / 4];
        int **A3 = new int[n * n / 4];
        int **A4 = new int[n * n / 4];
        int **B1 = new int[n * n / 4];
        int **B2 = new int[n * n / 4];
        int **B3 = new int[n * n / 4];
        int **B4 = new int[n * n / 4];

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        copyQuadrant(n, A, A1, 1);
        copyQuadrant(n, A, A2, 2);
        copyQuadrant(n, A, A3, 3);
        copyQuadrant(n, A, A4, 4);
        copyQuadrant(n, B, B1, 1);
        copyQuadrant(n, B, B2, 2);
        copyQuadrant(n, B, B3, 3);
        copyQuadrant(n, B, B4, 4);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);
        CALI_MARK_BEGIN(master_send);
        // send 7 (A & B) to children
        MPI_Send(A2, n * n / 4, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Send(B3, n * n / 4, MPI_INT, 1, 0, MPI_COMM_WORLD);

        MPI_Send(A1, n * n / 4, MPI_INT, 2, 0, MPI_COMM_WORLD);
        MPI_Send(B2, n * n / 4, MPI_INT, 2, 0, MPI_COMM_WORLD);

        MPI_Send(A2, n * n / 4, MPI_INT, 3, 0, MPI_COMM_WORLD);
        MPI_Send(B4, n * n / 4, MPI_INT, 3, 0, MPI_COMM_WORLD);

        MPI_Send(A3, n * n / 4, MPI_INT, 4, 0, MPI_COMM_WORLD);
        MPI_Send(B1, n * n / 4, MPI_INT, 4, 0, MPI_COMM_WORLD);

        MPI_Send(A4, n * n / 4, MPI_INT, 5, 0, MPI_COMM_WORLD);
        MPI_Send(B3, n * n / 4, MPI_INT, 5, 0, MPI_COMM_WORLD);

        MPI_Send(A3, n * n / 4, MPI_INT, 6, 0, MPI_COMM_WORLD);
        MPI_Send(B2, n * n / 4, MPI_INT, 6, 0, MPI_COMM_WORLD);

        MPI_Send(A4, n * n / 4, MPI_INT, 7, 0, MPI_COMM_WORLD);
        MPI_Send(B4, n * n / 4, MPI_INT, 7, 0, MPI_COMM_WORLD);
        CALI_MARK_END(master_send);
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(naive_recursion);
        // do ur own computation
        int *C1_part1 = naive_recursive_mult(n / 2, A1, B1);
        CALI_MARK_END(naive_recursion);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        // wait for C parts from all 7 children
        int *C1_part2;
        int *C2_part1;
        int *C2_part2;
        int *C3_part1;
        int *C3_part2;
        int *C4_part1;
        int *C4_part2;
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);
        CALI_MARK_BEGIN(master_recv);
        MPI_Recv(C1_part2, n * n / 4, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(C2_part1, n * n / 4, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(C2_part2, n * n / 4, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(C3_part1, n * n / 4, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(C3_part2, n * n / 4, MPI_INT, 5, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(C4_part1, n * n / 4, MPI_INT, 6, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(C4_part2, n * n / 4, MPI_INT, 7, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CALI_MARK_END(master_recv);
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);

        // add pairs and combine!
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(add_matrix);
        addM(n / 2, C1_part1, C1_part2);
        CALI_MARK_END(add_matrix);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(add_matrix);
        addM(n / 2, C2_part1, C2_part2);
        CALI_MARK_END(add_matrix);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(add_matrix);
        addM(n / 2, C3_part1, C3_part2);
        CALI_MARK_END(add_matrix);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(add_matrix);
        addM(n / 2, C4_part1, C4_part2);
        CALI_MARK_END(add_matrix);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        CALI_MARK_BEGIN(combine);
        for (int i = 0; i < n / 2; ++i){
            for (int j = 0; j < n / 2; ++j)
            {
                C[IDX(i, j, N)] = C1_part1[IDX(i, j, n / 2)];
                C[IDX(i, j + n / 2, N)] = C2_part1[IDX(i, j, n / 2)];
                C[IDX(i + n / 2, j, N)] = C3_part1[IDX(i, j, n / 2)];
                C[IDX(i + n / 2, j + n / 2, N)] = C4_part1[IDX(i, j, n / 2)];
            }
        }
        CALI_MARK_END(combine);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        // ! Cleanup
        // Warning, there's (3 Matrices) + (16 Quadrants) = 7 * N*N allocated data
        batchFreeM(
            new int *[]
            { A, B, C },
            3);
        batchFreeM(
            new int *[]
            { A1, A2, A3, A4, B1, B2, B3, B4,
                  C1_part1, C1_part2, C2_part1, C2_part2, C3_part1, C3_part2, C4_part1, C4_part2 },
            16);
    }
    else
    {
        int *quad_A;
        int *quad_B;
        int *quad_C;
        MPI_Status status;

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_recv);
        // receive from parent
        MPI_Recv(quad_A, n * n / 4, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(quad_B, n * n / 4, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        CALI_MARK_END(worker_recv);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        CALI_MARK_BEGIN(naive_recursion);
        // perform recursion
        quad_C = naive_recursive_mult(n / 2, quad_A, quad_B);
        CALI_MARK_END(naive_recursion);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(worker_send);
        // send back to parent
        MPI_Send(quad_C, n * n / 4, MPI_INT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(worker_send);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }

    MPI_Finalize();
}