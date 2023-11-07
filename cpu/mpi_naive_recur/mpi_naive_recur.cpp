#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int processors;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n; // matrix size
    if (argc == 2) {
      n = atoi(argv[1]);
    } else {
      printf("Please provide a matrix size\n");
      return 1;
    }

    // allocate matrices
    int a[n][n]; // matrix A
    int b[n][n]; // matrix B
    int c[n][n]; // result matrix C

    if (rank == 0) {
        // initialize matrices

        // partition matrices to 7 parts

        // send to children
        
        // wait for children

        // finalize things
    } else {
        // receive from parent 

        // perform basic strassen???

        // send back to parent
    }

    MPI_Finalize();
}