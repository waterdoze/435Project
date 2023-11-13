#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <bits/stdc++.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

int THREADS;
int BLOCKS;
int NUM_VALS;
int MATRIX_SIZE;

const char *data_init = "data_init";
const char *comm = "comm";
const char *comp = "comp";
const char *correctness = "correctness";
const char *strassen_time = "strassen_time";
const char *multi = "multi";

const char *slicing = "slicing";
const char *addsub = "addsub";
const char *combine = "combine";

const char *comm_small = "comm_small";
const char *comp_small = "comp_small";
const char *comm_large = "comm_large";
const char *comp_large = "comp_large";

const char *cudamemcpy = "cudamemcpy";
const char *cuda_naive_time = "cuda_naive_time";

cudaEvent_t strass_start, strass_stop, multi_start, multi_stop, h2d_start, h2d_stop, d2h_start, d2h_stop;

void print(int n, int **mat)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int *allocateMatrix(int n)
{
    int *data = (int *)malloc(n * n * sizeof(int));
    return data;
}

int **allocateMatrix2D(int n)
{
    int *data = (int *)malloc(n * n * sizeof(int));
    int **array = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++)
    {
        array[i] = &(data[n * i]);
    }
    return array;
}

void fillMatrix(int n, int *&mat)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mat[i * n + j] = rand() % 10 + 1;
        }
    }
}

void fillMatrix2D(int n, int **&mat)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mat[i][j] = rand() % 10 + 1;
        }
    }
}

int **getSlice(int n, int **mat, int offseti, int offsetj)
{
    int m = n / 2;
    int **slice = allocateMatrix2D(m);
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
    int **result = allocateMatrix2D(n);
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
    int **result = allocateMatrix2D(n);

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

void freeMatrix(int n, int *mat)
{
    free(mat);
}

void freeMatrix2D(int n, int **mat)
{
    free(mat[0]);
    free(mat);
}

int **naive(int n, int **mat1, int **mat2)
{
    int **prod = allocateMatrix2D(n);

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

__global__ void multiply(int *a, int *b, int *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    c[row * n + col] = 0;
    for (int k = 0; k < n; k++)
    {
        // Accumulate results for a single element
        c[row * n + col] += a[row * n + k] * b[k * n + col];
    }
}

int **cudaNaive(int n, int **mat1, int **mat2)
{
    int *h_mat1 = allocateMatrix(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            h_mat1[i * n + j] = mat1[i][j];
        }
    }

    int *h_mat2 = allocateMatrix(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            h_mat2[i * n + j] = mat2[i][j];
        }
    }

    int *h_product = allocateMatrix(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            h_product[i * n + j] = 0;
        }
    }

    size_t bytes = n * n * sizeof(int);

    int *d_mat1, *d_mat2, *d_product;

    cudaMalloc(&d_mat1, bytes);
    cudaMalloc(&d_mat2, bytes);
    cudaMalloc(&d_product, bytes);

    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_stop);

    cudaEventRecord(h2d_start);
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(cudamemcpy);
    cudaMemcpy(d_mat1, h_mat1, bytes, cudaMemcpyHostToDevice);
    CALI_MARK_END(cudamemcpy);
    CALI_MARK_BEGIN(cudamemcpy);
    cudaMemcpy(d_mat2, h_mat2, bytes, cudaMemcpyHostToDevice);
    CALI_MARK_END(cudamemcpy);
    CALI_MARK_BEGIN(cudamemcpy);
    cudaMemcpy(d_product, h_product, bytes, cudaMemcpyHostToDevice);
    CALI_MARK_END(cudamemcpy);
    CALI_MARK_END(comm);
    cudaEventRecord(h2d_stop);

    int threads = THREADS;
    int blocks = (n + THREADS - 1) / THREADS;
    dim3 gridSize(blocks, blocks, 1);
    dim3 blockSize(threads, threads, 1);

    cudaEventCreate(&multi_start);
    cudaEventCreate(&multi_stop);

    cudaEventRecord(multi_start);
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(multi);
    multiply<<<gridSize, blockSize>>>(d_mat1, d_mat2, d_product, n);
    cudaDeviceSynchronize();
    CALI_MARK_END(multi);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    cudaEventRecord(multi_stop);

    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_stop);

    cudaEventRecord(d2h_start);
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(cudamemcpy);
    cudaMemcpy(h_product, d_product, bytes, cudaMemcpyDeviceToHost);
    CALI_MARK_END(cudamemcpy);
    CALI_MARK_END(comm);
    cudaEventRecord(d2h_stop);

    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_product);

    freeMatrix(n, h_mat1);
    freeMatrix(n, h_mat2);

    int **product = allocateMatrix2D(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            product[i][j] = h_product[i * n + j];
        }
    }
    return product;
}

int **strassen(int n, int **mat1, int **mat2)
{

    if (n <= 32)
    {
        return naive(n, mat1, mat2);
    }

    int m = n / 2;
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(slicing);
    int **a = getSlice(n, mat1, 0, 0);
    int **b = getSlice(n, mat1, 0, m);
    int **c = getSlice(n, mat1, m, 0);
    int **d = getSlice(n, mat1, m, m);
    int **e = getSlice(n, mat2, 0, 0);
    int **f = getSlice(n, mat2, 0, m);
    int **g = getSlice(n, mat2, m, 0);
    int **h = getSlice(n, mat2, m, m);
    CALI_MARK_END(slicing);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **bds = addMatrices(m, b, d, false);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **gha = addMatrices(m, g, h, true);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    int **s1 = cudaNaive(m, bds, gha);
    freeMatrix2D(m, bds);
    freeMatrix2D(m, gha);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **ada = addMatrices(m, a, d, true);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **eha = addMatrices(m, e, h, true);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    int **s2 = cudaNaive(m, ada, eha);
    freeMatrix2D(m, ada);
    freeMatrix2D(m, eha);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **acs = addMatrices(m, a, c, false);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **efa = addMatrices(m, e, f, true);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    int **s3 = cudaNaive(m, acs, efa);
    freeMatrix2D(m, acs);
    freeMatrix2D(m, efa);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **aba = addMatrices(m, a, b, true);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    int **s4 = cudaNaive(m, aba, h);
    freeMatrix2D(m, aba);
    freeMatrix2D(m, b);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **fhs = addMatrices(m, f, h, false);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    int **s5 = cudaNaive(m, a, fhs);
    freeMatrix2D(m, fhs);
    freeMatrix2D(m, a);
    freeMatrix2D(m, f);
    freeMatrix2D(m, h);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **ges = addMatrices(m, g, e, false);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    int **s6 = cudaNaive(m, d, ges);
    freeMatrix2D(m, ges);
    freeMatrix2D(m, g);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **cda = addMatrices(m, c, d, true);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    int **s7 = cudaNaive(m, cda, e);
    freeMatrix2D(m, cda);
    freeMatrix2D(m, c);
    freeMatrix2D(m, d);
    freeMatrix2D(m, e);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **s1s2a = addMatrices(m, s1, s2, true);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **s6s4s = addMatrices(m, s6, s4, false);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **c11 = addMatrices(m, s1s2a, s6s4s, true);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    freeMatrix2D(m, s1s2a);
    freeMatrix2D(m, s6s4s);
    freeMatrix2D(m, s1);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **c12 = addMatrices(m, s4, s5, true);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    freeMatrix2D(m, s4);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **c21 = addMatrices(m, s6, s7, true);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    freeMatrix2D(m, s6);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **s2s3s = addMatrices(m, s2, s3, false);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **s5s7s = addMatrices(m, s5, s7, false);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(addsub);
    int **c22 = addMatrices(m, s2s3s, s5s7s, true);
    CALI_MARK_END(addsub);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    freeMatrix2D(m, s2s3s);
    freeMatrix2D(m, s5s7s);
    freeMatrix2D(m, s2);
    freeMatrix2D(m, s3);
    freeMatrix2D(m, s5);
    freeMatrix2D(m, s7);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    CALI_MARK_BEGIN(combine);
    int **prod = combineMatrices(m, c11, c12, c21, c22);
    CALI_MARK_END(combine);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    freeMatrix2D(m, c11);
    freeMatrix2D(m, c12);
    freeMatrix2D(m, c21);
    freeMatrix2D(m, c22);

    // CALI_MARK_BEGIN(comp);
    return prod;
}

int main(int argc, char *argv[])
{

    cali::ConfigManager mgr;
    mgr.start();

    int n;
    n = atoi(argv[2]);
    THREADS = atoi(argv[1]);
    printf("Threads: %d\n", THREADS);
    printf("Matrix Size: %d\n", n);
    int **mat1 = allocateMatrix2D(n);
    int **mat2 = allocateMatrix2D(n);
    CALI_MARK_BEGIN(data_init);
    fillMatrix2D(n, mat1);
    fillMatrix2D(n, mat2);
    CALI_MARK_END(data_init);

    cudaEventCreate(&strass_start);
    cudaEventCreate(&strass_stop);
    clock_t start, end;
    start = clock();
    cudaEventRecord(strass_start);
    int **prod = strassen(n, mat1, mat2);
    cudaEventRecord(strass_stop);
    end = clock();
    double time = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Parallel Strassen Runtime (CUDA): " << time << " seconds\n";

    cudaEventSynchronize(multi_stop);
    cudaEventSynchronize(h2d_stop);
    cudaEventSynchronize(d2h_stop);
    cudaEventSynchronize(strass_stop);

    float multiply_time, h2d_time, d2h_time, strass_time;
    cudaEventElapsedTime(&multiply_time, multi_start, multi_stop);
    cudaEventElapsedTime(&h2d_time, h2d_start, h2d_stop);
    cudaEventElapsedTime(&d2h_time, d2h_start, d2h_stop);
    cudaEventElapsedTime(&strass_time, strass_start, strass_stop);

    printf("Multiply Time (ms): %f\n", multiply_time);
    printf("H2D Time (ms): %f\n", h2d_time);
    printf("D2H Time (ms): %f\n", d2h_time);
    printf("Strassen Time (ms): %f\n", strass_time);

    // verify correctness
    CALI_MARK_BEGIN(correctness);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float temp = 0;
            for (int k = 0; k < n; k++)
            {
                temp += mat1[i][k] * mat2[k][j];
            }
            if (prod[i][j] != temp)
            {
                printf("Incorrect at %d, %d\n", i, j);
                printf("Expected: %f, Actual: %f\n", temp, prod[i][j]);
                return 0;
            }
        }
    }
    CALI_MARK_END(correctness);
    printf("Verification Successful\n");

    adiak::init(NULL);
    adiak::launchdate();                                           // launch date of the job
    adiak::libraries();                                            // Libraries used
    adiak::cmdline();                                              // Command line used to launch the job
    adiak::clustername();                                          // Name of the cluster
    adiak::value("Algorithm", "CUDA Naive Matrix Multiplication"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA");                      // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                               // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));                   // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", MATRIX_SIZE);                        // The number of elements in input dataset (1000)
    // adiak::value("InputType", inputType);                        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    // adiak::value("num_procs", size);                        // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS);   // The number of CUDA blocks
    adiak::value("group_num", 8);         // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online");

    return 0;
}
