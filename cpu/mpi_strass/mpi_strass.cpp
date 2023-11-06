#include <mpi.h>
#include <bits/stdc++.h>

using namespace std;

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
    int **s1 = strassen(m, a, fh_sub);
    freeMatrix(m, fh_sub);

    int **ab_add = addMatrices(m, a, b, true); // A11 + A12
    int **s2 = strassen(m, ab_add, h);
    freeMatrix(m, ab_add);

    int **cd_add = addMatrices(m, c, d, true); // A21 + A22
    int **s3 = strassen(m, cd_add, e);
    freeMatrix(m, cd_add);

    int **ge_sub = addMatrices(m, g, e, false); // B21 - B11
    int **s4 = strassen(m, d, ge_sub);
    freeMatrix(m, ge_sub);

    int **ad_add = addMatrices(m, a, d, true); // A11 + A22
    int **eh_add = addMatrices(m, e, h, true); // B11 + B22
    int **s5 = strassen(m, ad_add, eh_add);
    freeMatrix(m, ad_add);
    freeMatrix(m, eh_add);

    int **bd_sub = addMatrices(m, b, d, false); // B12 - B22
    int **gh_add = addMatrices(m, g, h, true);  // B21 + B22
    int **s6 = strassen(m, bd_sub, gh_add);
    freeMatrix(m, bd_sub);
    freeMatrix(m, gh_add);

    int **ac_sub = addMatrices(m, a, c, false); // A11 - A21
    int **ef_add = addMatrices(m, e, f, true);  // B11 + B12
    int **s7 = strassen(m, ac_sub, ef_add);
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

    int **a = getSlice(n, mat1, 0, 0); // A11
    int **b = getSlice(n, mat1, 0, m); // A12
    int **c = getSlice(n, mat1, m, 0); // A21
    int **d = getSlice(n, mat1, m, m); // A22
    int **e = getSlice(n, mat2, 0, 0); // B11
    int **f = getSlice(n, mat2, 0, m); // B12
    int **g = getSlice(n, mat2, m, 0); // B21
    int **h = getSlice(n, mat2, m, m); // B22

    int **s1 = allocateMatrix(m);
    int **s2 = allocateMatrix(m);
    int **s3 = allocateMatrix(m);
    int **s4 = allocateMatrix(m);
    int **s5 = allocateMatrix(m);
    int **s6 = allocateMatrix(m);
    int **s7 = allocateMatrix(m);

    if (rank == 0)
    {
        MPI_Recv(&(s1[0][0]), m * m, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s2[0][0]), m * m, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s3[0][0]), m * m, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s4[0][0]), m * m, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s5[0][0]), m * m, MPI_INT, 5, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s6[0][0]), m * m, MPI_INT, 6, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s7[0][0]), m * m, MPI_INT, 7, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank == 1)
    {
        int **fh_sub = addMatrices(m, f, h, false); // B12 - B22
        s1 = strassen(m, a, fh_sub);
        freeMatrix(m, fh_sub);
        MPI_Send(&(s1[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 2)
    {
        int **ab_add = addMatrices(m, a, b, true); // A11 + A12
        s2 = strassen(m, ab_add, h);
        freeMatrix(m, ab_add);
        MPI_Send(&(s2[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 3)
    {
        int **cd_add = addMatrices(m, c, d, true); // A21 + A22
        s3 = strassen(m, cd_add, e);
        freeMatrix(m, cd_add);
        MPI_Send(&(s3[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 4)
    {
        int **ge_sub = addMatrices(m, g, e, false); // B21 - B11
        s4 = strassen(m, d, ge_sub);
        freeMatrix(m, ge_sub);
        MPI_Send(&(s4[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 5)
    {
        int **ad_add = addMatrices(m, a, d, true); // A11 + A22
        int **eh_add = addMatrices(m, e, h, true); // B11 + B22
        s5 = strassen(m, ad_add, eh_add);
        freeMatrix(m, ad_add);
        freeMatrix(m, eh_add);
        MPI_Send(&(s5[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 6)
    {
        int **bd_sub = addMatrices(m, b, d, false); // B12 - B22
        int **gh_add = addMatrices(m, g, h, true);  // B21 + B22
        s6 = strassen(m, bd_sub, gh_add);
        freeMatrix(m, bd_sub);
        freeMatrix(m, gh_add);
        MPI_Send(&(s6[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 7)
    {
        int **ac_sub = addMatrices(m, a, c, false); // A11 - A21
        int **ef_add = addMatrices(m, e, f, true);  // B11 + B12
        s7 = strassen(m, ac_sub, ef_add);
        freeMatrix(m, ac_sub);
        freeMatrix(m, ef_add);
        MPI_Send(&(s7[0][0]), m * m, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    freeMatrix(m, a);
    freeMatrix(m, b);
    freeMatrix(m, c);
    freeMatrix(m, d);
    freeMatrix(m, e);
    freeMatrix(m, f);
    freeMatrix(m, g);
    freeMatrix(m, h);

    MPI_Barrier(MPI_COMM_WORLD);

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

        prod = combineMatrices(m, c11, c12, c21, c22);

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

bool check(int n, int **prod1, int **prod2)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (prod1[i][j] != prod2[i][j])
                return false;
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    int p_rank;
    int num_process;

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    {
        printf("MPI-INIT Failed\n");
        return 0;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);

    int n;
    n = atoi(argv[1]);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int **mat1 = allocateMatrix(n);
    int **mat2 = allocateMatrix(n);

    if (p_rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                mat1[i][j] = rand() % 10 + 1;
                mat2[i][j] = rand() % 10 + 1;
            }
        }
        // print(n, mat1);
        // print(n, mat2);
    }

    MPI_Bcast(&(mat1[0][0]), n * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(mat2[0][0]), n * n, MPI_INT, 0, MPI_COMM_WORLD);

    double startTime = MPI_Wtime();

    int **prod;
    strassen(n, mat1, mat2, prod, p_rank);

    double endTime = MPI_Wtime();

    if (p_rank == 0)
    {
        printf("\nParallel Strassen Runtime (MPI): ");
        printf("%.5f\n\n", endTime - startTime);
        print(n, prod);
        int **naive_prod = naive(n, mat1, mat2);
        if (check(n, prod, naive_prod))
            cout << "Verification Passed!" << endl;
        else
            cout << "Error." << endl;
    }

    MPI_Finalize();

    return 0;
}