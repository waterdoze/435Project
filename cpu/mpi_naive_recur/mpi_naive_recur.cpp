#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int** allocateM(int size) {
    int** matrix = new int*[size];
    for (int i = 0; i < size; ++i)
        matrix[i] = new int[size];
    return matrix;
}

void freeM(int size, int** matrix) {
    for (int i = 0; i < size; ++i)
        delete[] matrix[i];
    delete[] matrix;
}

void copyQuadrant(int src_size, int** src, int** dest, int quadrant) {
    // quad =
    // 1 | 2
    // 3 | 4
    int dest_size = src_size / 2;
    int row_offset = (quadrant == 1 || quadrant == 2) ? 0 : dest_size;
    int col_offset = (quadrant == 1 || quadrant == 3) ? 0 : dest_size;
    for (int i = 0; i < dest_size; ++i)
        for (int j = 0; j < dest_size; ++j)
            dest[i][j] = src[i + row_offset][j + col_offset];
}

void addM(int size, int** a, int** b) {
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; j++)
            a[i][j] += b[i][j];
}

int** naive_recursive_mult(int size, int** A, int** B) {
    // allocate C
    // base case
    // allocate & init 8 quads
    // multiply quads
    // add quad pairs
    // combine to C
    // free 8 quads
    // return C
}

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

        // split into 8 pieces

        // send 7 to children
        
        // wait for children

        // finalize things
    } else {
        // receive from parent 

        // perform recursion

        // send back to parent
    }

    MPI_Finalize();
}