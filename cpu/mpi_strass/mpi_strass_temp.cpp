#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;

// Define a function to add two matrices
void matrixAdd(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

// Define a function to subtract two matrices
void matrixSub(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

// Define a function to split a matrix into four submatrices
void splitMatrix(const vector<vector<int>> &A, vector<vector<int>> &A11, vector<vector<int>> &A12,
                 vector<vector<int>> &A21, vector<vector<int>> &A22, int size)
{
    int newSize = size / 2;
    for (int i = 0; i < newSize; i++)
    {
        for (int j = 0; j < newSize; j++)
        {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];
        }
    }
}

// Define a function to combine four submatrices into a single matrix
void combineMatrices(vector<vector<int>> &C, const vector<vector<int>> &C11, const vector<vector<int>> &C12,
                     const vector<vector<int>> &C21, const vector<vector<int>> &C22, int size)
{
    int newSize = size / 2;
    for (int i = 0; i < newSize; i++)
    {
        for (int j = 0; j < newSize; j++)
        {
            C[i][j] = C11[i][j];
            C[i][j + newSize] = C12[i][j];
            C[i + newSize][j] = C21[i][j];
            C[i + newSize][j + newSize] = C22[i][j];
        }
    }
}

// Strassen's matrix multiplication algorithm
void strassen(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int size)
{
    if (size == 1)
    {
        C[0][0] = A[0][0] * B[0][0];
        return;
    }

    int newSize = size / 2;

    // Create submatrices
    vector<vector<int>> A11(newSize, vector<int>(newSize));
    vector<vector<int>> A12(newSize, vector<int>(newSize));
    vector<vector<int>> A21(newSize, vector<int>(newSize));
    vector<vector<int>> A22(newSize, vector<int>(newSize));
    vector<vector<int>> B11(newSize, vector<int>(newSize));
    vector<vector<int>> B12(newSize, vector<int>(newSize));
    vector<vector<int>> B21(newSize, vector<int>(newSize));
    vector<vector<int>> B22(newSize, vector<int>(newSize));

    splitMatrix(A, A11, A12, A21, A22, size);
    splitMatrix(B, B11, B12, B21, B22, size);

    // Calculate intermediate matrices
    vector<vector<int>> M1(newSize, vector<int>(newSize));
    vector<vector<int>> M2(newSize, vector<int>(newSize));
    vector<vector<int>> M3(newSize, vector<int>(newSize));
    vector<vector<int>> M4(newSize, vector<int>(newSize));
    vector<vector<int>> M5(newSize, vector<int>(newSize));
    vector<vector<int>> M6(newSize, vector<int>(newSize));
    vector<vector<int>> M7(newSize, vector<int>(newSize));

    vector<vector<int>> C11(newSize, vector<int>(newSize));
    vector<vector<int>> C12(newSize, vector<int>(newSize));
    vector<vector<int>> C21(newSize, vector<int>(newSize));
    vector<vector<int>> C22(newSize, vector<int>(newSize));

    vector<vector<int>> AResult(newSize, vector<int>(newSize));
    vector<vector<int>> BResult(newSize, vector<int>(newSize));

    // Calculate M1, M2, M3, M4, M5, M6, M7
    matrixAdd(A11, A22, AResult, newSize);
    matrixAdd(B11, B22, BResult, newSize);
    strassen(AResult, BResult, M1, newSize);

    matrixAdd(A21, A22, AResult, newSize);
    strassen(AResult, B11, M2, newSize);

    matrixSub(B12, B22, BResult, newSize);
    strassen(A11, BResult, M3, newSize);

    matrixSub(B21, B11, BResult, newSize);
    strassen(A22, BResult, M4, newSize);

    matrixAdd(A11, A12, AResult, newSize);
    strassen(AResult, B22, M5, newSize);

    matrixSub(A21, A11, AResult, newSize);
    matrixAdd(B11, B12, BResult, newSize);
    strassen(AResult, BResult, M6, newSize);

    matrixSub(A12, A22, AResult, newSize);
    matrixAdd(B21, B22, BResult, newSize);
    strassen(AResult, BResult, M7, newSize);

    // Calculate C11, C12, C21, C22
    matrixAdd(M1, M4, AResult, newSize);
    matrixSub(M7, M5, BResult, newSize);
    matrixAdd(AResult, BResult, C11, newSize);

    matrixAdd(M3, M5, C12, newSize);

    matrixAdd(M2, M4, C21, newSize);

    matrixAdd(M1, M3, AResult, newSize);
    matrixSub(M2, M6, BResult, newSize);
    matrixAdd(AResult, BResult, C22, newSize);

    // Combine submatrices to get the result
    combineMatrices(C, C11, C12, C21, C22, size);
}

// Helper function to print a matrix
void printMatrix(const vector<vector<int>> &matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            cout << matrix[i][j] << " ";
    }
    cout << endl;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Size of the matrices (must be a power of 2)
    const int n = 8;
    const int matrixSize = 1 << n;

    if (size != 4)
    {
        if (rank == 0)
        {
            cout << "This program must be run with 4 processes." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (matrixSize % size != 0)
    {
        if (rank == 0)
        {
            cout << "Matrix size must be divisible by the number of processes." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    vector<vector<int>> A(matrixSize, vector<int>(matrixSize));
    vector<vector<int>> B(matrixSize, vector<int>(matrixSize));
    vector<vector<int>> C(matrixSize, vector<int>(matrixSize));

    // Initialize matrices A and B with random values
    if (rank == 0)
    {
        srand(static_cast<unsigned int>(time(NULL)));
        for (int i = 0; i < matrixSize; i++)
        {
            for (int j = 0; j < matrixSize; j++)
            {
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;
            }
        }
    }

    // Scatter the matrices A and B to all processes
    MPI_Bcast(&A[0][0], matrixSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B[0][0], matrixSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);

    const int blockSize = matrixSize / size;
    vector<vector<int>> localA(blockSize, vector<int>(matrixSize));
    vector<vector<int>> localB(blockSize, vector<int>(matrixSize));
    vector<vector<int>> localC(blockSize, vector<int>(matrixSize));

    // Scatter blocks of A and B to all processes
    MPI_Scatter(&A[0][0], blockSize * matrixSize, MPI_INT, &localA[0][0], blockSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(&B[0][0], blockSize * matrixSize, MPI_INT, &localB[0][0], blockSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform local matrix multiplication using Strassen's algorithm
    strassen(localA, localB, localC, blockSize);

    // Gather the results from all processes
    MPI_Gather(&localC[0][0], blockSize * matrixSize, MPI_INT, &C[0][0], blockSize * matrixSize, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "Matrix A:" << endl;
        printMatrix(A, matrixSize);

        cout << "Matrix B:" << endl;
        printMatrix(B, matrixSize);

        cout << "Result C:" << endl;
        printMatrix(C, matrixSize);
    }

    MPI_Finalize();

    return 0;
}
