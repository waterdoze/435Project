
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stddef.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
    int i, j, k, rank, size, sum = 0;
    int N = atoi(argv[1]);

    int a[N][N]; // Matrix A
    int b[N][N]; // Matrix B
    int c[N][N]; // Result Matrix C

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int aRows[N / size][N];   // hold rows of matrix A to be scattered
    int cCols[N / size][N];   // hold columns of matrix C to be gathered
    int block = N * N / size; // number of elements in each block

    if (rank == 0)
    {
        printf("Matrix Size: %d\n", N);
        printf("Number of Processes: %d\n", size);

        // initialize matrices
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                a[i][j] = rand() % 10 + 1;
                b[i][j] = rand() % 10 + 1;
            }
        }

        // for (i = 0; i < N; i++)
        // {
        //     for (j = 0; j < N; j++)
        //     {
        //         printf(" %d", a[i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");

        // for (i = 0; i < N; i++)
        // {
        //     for (j = 0; j < N; j++)
        //     {
        //         printf(" %d", b[i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");
    }

    // scatter rows of first matrix to different processes
    MPI_Scatter(a, block, MPI_INT, aRows, block, MPI_INT, 0, MPI_COMM_WORLD);

    // broadcast second matrix to all processes
    MPI_Bcast(b, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // wait for all processes to finish broadcasting
    MPI_Barrier(MPI_COMM_WORLD);

    // perform vector multiplication by all processes
    for (i = 0; i < N / size; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < N; k++)
            {
                sum += aRows[i][k] * b[k][j];
            }
            cCols[i][j] = sum;
            sum = 0;
        }
    }

    // gather columns of matrix C to root process
    MPI_Gather(cCols, block, MPI_INT, c, block, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    if (rank == 0)
    {
        // for (i = 0; i < N; i++)
        // {
        //     for (j = 0; j < N; j++)
        //     {
        //         printf(" %d", c[i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");

        // check if result is correct
        int correct[N][N];
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                correct[i][j] = 0;
                for (int k = 0; k < N; k++)
                {
                    correct[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                // printf(" %d", correct[i][j]);
                if (correct[i][j] != c[i][j])
                {
                    printf("Error at %d, %d\n", i, j);
                    return 1;
                }
            }
        }
        printf("Verification Passed!\n");
    }
}