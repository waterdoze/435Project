
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include "mpi.h"

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int main(int argc, char *argv[])
{

    CALI_CXX_MARK_FUNCTION;

    int i, j, k, rank, size, sum = 0;
    int n; // Matrix size
    if(argc == 2) {
        n = atoi(argv[1]);
    }
    else {
        printf("Please provide a matrix size\n");
        return 1;
    }

    int a[n][n]; // Matrix A
    int b[n][n]; // Matrix B
    int c[n][n]; // Result Matrix C

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size); // number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // rank of each process

    int aRows[n / size][n];   // hold rows of matrix A to be scattered
    int cCols[n / size][n];   // hold columns of matrix C to be gathered
    int block = n * n / size; // number of elements in each block

    /* Define Caliper region names */
    const char* data_init = "data_init";
    const char* comm = "comm";
    const char* comm_small = "comm_small";
    const char* comm_large = "comm_large";
    const char* comp = "comp";
    const char* comp_small = "comp_small";
    const char* comp_large = "comp_large";
    const char* correctness = "correctness";

    const char* scatter = "scatter";
    const char* bcast = "bcast";
    const char* barrier = "barrier";
    const char* gather = "gather";

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);

    cali::ConfigManager mgr;
    mgr.start();

    if (rank == 0)
    {
        printf("Matrix Size: %d\n", n);
        printf("Number of Processes: %d\n", size);

        CALI_MARK_BEGIN(data_init);
        // initialize matrices
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                a[i][j] = rand() % 10 + 1;
                b[i][j] = rand() % 10 + 1;
            }
        }
        CALI_MARK_END(data_init);

        // for (i = 0; i < n; i++)
        // {
        //     for (j = 0; j < n; j++)
        //     {
        //         printf(" %d", a[i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");

        // for (i = 0; i < n; i++)
        // {
        //     for (j = 0; j < n; j++)
        //     {
        //         printf(" %d", b[i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");
    }

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(scatter);
    // scatter rows of first matrix to different processes
    MPI_Scatter(a, block, MPI_INT, aRows, block, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(scatter);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(bcast);
    // broadcast second matrix to all processes
    MPI_Bcast(b, n * n, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(bcast);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(barrier);
    // wait for all processes to finish broadcasting
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(barrier);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    // perform vector multiplication by all processes
    for (i = 0; i < n / size; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < n; k++)
            {
                sum += aRows[i][k] * b[k][j];
            }
            cCols[i][j] = sum;
            sum = 0;
        }
    }
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(gather);
    // gather columns of matrix C to root process
    MPI_Gather(cCols, block, MPI_INT, c, block, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(gather);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(barrier);
    CALI_MARK_END(comm);

    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);


    adiak::init(NULL);
    adiak::launchdate();                                         // launch date of the job
    adiak::libraries();                                          // Libraries used
    adiak::cmdline();                                            // Command line used to launch the job
    adiak::clustername();                                        // Name of the cluster
    adiak::value("Algorithm", "MPI Naive Matrix Multiplication");// The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");          // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                          // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));              // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n);                        // The number of elements in input dataset (1000)
    // adiak::value("InputType", inputType);                        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", size);                        // The number of processors (MPI ranks)
    // adiak::value("num_threads", num_threads);                    // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", num_blocks);                      // The number of CUDA blocks
    adiak::value("group_num", 8);                     // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    if (rank == 0)
    {
        // Print out the result matrix
        // for (i = 0; i < n; i++)
        // {
        //     for (j = 0; j < n; j++)
        //     {
        //         printf(" %d", c[i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");

        CALI_MARK_BEGIN(correctness);
        // check if result is correct
        int correct[n][n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                correct[i][j] = 0;
                for (int k = 0; k < n; k++)
                {
                    correct[i][j] += a[i][k] * b[k][j];
                }
                if(correct[i][j] != c[i][j]) {
                    printf("Error at %d, %d\n", i, j);
                    printf("Expected: %d, Actual: %d\n", correct[i][j], c[i][j])
                    return 1;
                }
            }
        }

        // for (int i = 0; i < n; i++)
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         // printf(" %d", correct[i][j]);
        //         if (correct[i][j] != c[i][j])
        //         {
        //             printf("Error at %d, %d\n", i, j);
        //             return 1;
        //         }
        //     }
        // }
        CALI_MARK_END(correctness);
        printf("Verification Passed!\n");
    }

    mgr.stop();
    mgr.flush();

    MPI_Finalize();
}