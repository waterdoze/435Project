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

void batchFreeM(int size, int** matrix[], int num_matrices) {
    for (int i = 0; i < num_matrices; ++i)
        freeM(size, matrix[i]);
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
    int qsize = size / 2;

    // allocate C
    int** C = allocateM(size);

    // base case
    if (size == 1) {
        C[0][0] = A[0][0] * B[0][0];
        return C;
    }

    // allocate & init 8 quads
    int** A1 = allocateM(qsize);
    int** A2 = allocateM(qsize);
    int** A3 = allocateM(qsize);
    int** A4 = allocateM(qsize);
    int** B1 = allocateM(qsize);
    int** B2 = allocateM(qsize);
    int** B3 = allocateM(qsize);
    int** B4 = allocateM(qsize);

    copyQuadrant(size, A, A1, 1);
    copyQuadrant(size, A, A2, 2);
    copyQuadrant(size, A, A3, 3);
    copyQuadrant(size, A, A4, 4);
    copyQuadrant(size, B, B1, 1);
    copyQuadrant(size, B, B2, 2);
    copyQuadrant(size, B, B3, 3);
    copyQuadrant(size, B, B4, 4);

    // multiply quads
    int** C1_part1 = naive_recursive_mult(qsize, A1, B1);
    int** C1_part2 = naive_recursive_mult(qsize, A2, B3);
    int** C2_part1 = naive_recursive_mult(qsize, A1, B2);
    int** C2_part2 = naive_recursive_mult(qsize, A2, B4);
    int** C3_part1 = naive_recursive_mult(qsize, A3, B1);
    int** C3_part2 = naive_recursive_mult(qsize, A4, B3);
    int** C4_part1 = naive_recursive_mult(qsize, A3, B2);
    int** C4_part2 = naive_recursive_mult(qsize, A4, B4);

    // add quad pairs
    addM(qsize, C1_part1, C1_part2);
    addM(qsize, C2_part1, C2_part2);
    addM(qsize, C3_part1, C3_part2);
    addM(qsize, C4_part1, C4_part2);

    // combine to C
    for (int i = 0; i < qsize; ++i)
        for (int j = 0; j < qsize; ++j) {
            C[i][j] = C1_part1[i][j];
            C[i][j + qsize] = C2_part1[i][j];
            C[i + qsize][j] = C3_part1[i][j];
            C[i + qsize][j + qsize] = C4_part1[i][j];
        }

    // free allocated A & B quads & returned C parts
    // ! Could fail idk, just check if compilation error happens
    // TODO: if batchFreeM works fine, delete this comment. freeM(qsize, A1); freeM(qsize, A2); freeM(qsize, A3); freeM(qsize, A4); freeM(qsize, B1); freeM(qsize, B2); freeM(qsize, B3); freeM(qsize, B4); freeM(qsize, C1_part1); freeM(qsize, C1_part2); freeM(qsize, C2_part1); freeM(qsize, C2_part2); freeM(qsize, C3_part1); freeM(qsize, C3_part2); freeM(qsize, C4_part1); freeM(qsize, C4_part2);
    batchFreeM(qsize, new int**[] {
        A1, A2, A3, A4, B1, B2, B3, B4, 
        C1_part1, C1_part2, C2_part1, C2_part2, C3_part1, C3_part2, C4_part1, C4_part2
    }, 16);

    // return C
    return C;
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
    int** a; // matrix A
    int** b; // matrix B
    int** c; // result matrix C
    int individual_matrix_size = n / 2;

    if (rank == 0) {
        // initialize matrices
        a = allocateM(n);
        b = allocateM(n);
        c = allocateM(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; ++j) {
                a[i][j] = rand() % 10 + 1;
                b[i][j] = rand() % 10 + 1;
            }
        }

        // split into 8 pieces

        // send 7 to children
        
        // wait for children

        // combine and finalize

        freeM(n, a);
        freeM(n, b);
        freeM(n, c);
    } else {
        // receive from parent 

        // perform recursion

        // send back to parent
    }

    MPI_Finalize();
}