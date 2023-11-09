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

void print(int n, int** mat)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int* allocateMatrix(int n)
{
    int* data = (int*)malloc(n * n * sizeof(int));
    return data;
}

int** allocateMatrix2D(int n)
{
    int* data = (int*)malloc(n * n * sizeof(int));
    int** array = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++)
    {
        array[i] = &(data[n * i]);
    }
    return array;
}

void fillMatrix(int n, int*& mat)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i * n + j] = rand() % 10 + 1;
        }
    }
}

void fillMatrix2D(int n, int** &mat)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = rand() % 10 + 1;
        }
    }
}

int** getSlice(int n, int** mat, int offseti, int offsetj)
{
    int m = n / 2;
    int** slice = allocateMatrix2D(m);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            slice[i][j] = mat[offseti + i][offsetj + j];
        }
    }
    return slice;
}

int** addMatrices(int n, int** mat1, int** mat2, bool add)
{
    int** result = allocateMatrix2D(n);
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

int** combineMatrices(int m, int** c11, int** c12, int** c21, int** c22)
{
    int n = 2 * m;
    int** result = allocateMatrix2D(n);

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

void freeMatrix(int n, int* mat)
{
    free(mat);
}

void freeMatrix2D(int n, int** mat)
{
    free(mat[0]);
    free(mat);
}

int** naive(int n, int** mat1, int** mat2)
{
    int** prod = allocateMatrix2D(n);

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

__global__ void multiply(int* a, int* b, int* c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    c[row * N + col] = 0;
    for (int k = 0; k < N; k++) {
            // Accumulate results for a single element
        c[row * N + col] += a[row * N + k] * b[k * N + col];
    }
}

int** cudaNaive(int n, int** mat1, int** mat2)
{
    int* h_mat1 = allocateMatrix(n);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            h_mat1[i*n + j] = mat1[i][j];
        }
    }

    int* h_mat2 = allocateMatrix(n);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            h_mat2[i*n + j] = mat2[i][j];
        }
    }

    int* h_product = allocateMatrix(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_product[i * n + j] = 0;
        }
    }

    size_t bytes = n * n * sizeof(int);

    int *d_mat1, *d_mat2, *d_product;

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(host2device);
    cudaMalloc(&d_mat1, bytes);
    cudaMalloc(&d_mat2, bytes);
    cudaMalloc(&d_product, bytes);
    
    cudaMemcpy(d_mat1, h_mat1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, h_mat2, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_product, h_product, bytes, cudaMemcpyHostToDevice);
    CALI_MARK_END(host2device);
    CALI_MARK_END(comm);

    int threads = THREADS;
    int blocks = (MATRIX_SIZE + THREADS - 1) / THREADS;
    dim3 gridSize(blocks, blocks, 1);
    dim3 blockSize(threads, threads, 1);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(naive_multi);
    multiply<<<gridSize, blockSize>>>(d_mat1, d_mat2, d_product, n);
    cudaDeviceSynchronize();
    CALI_MARK_END(naive_multi);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(device2host);
    cudaMemcpy(h_product, d_product, bytes, cudaMemcpyDeviceToHost);
    CALI_MARK_END(device2host);
    CALI_MARK_END(comm);

    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_product);

    freeMatrix(n, h_mat1);
    freeMatrix(n, h_mat2);

    int** product = allocateMatrix2D(n);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            product[i][j] = h_product[i*n + j];
        }
    }
    return product;
}


int** strassen(int n, int** mat1, int** mat2)
{
    
    if (n <= 32)
    {
        return naive(n, mat1, mat2);
    }

    int m = n / 2;
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(slicing);
    int** a = getSlice(n, mat1, 0, 0);
    int** b = getSlice(n, mat1, 0, m);
    int** c = getSlice(n, mat1, m, 0);
    int** d = getSlice(n, mat1, m, m);
    int** e = getSlice(n, mat2, 0, 0);
    int** f = getSlice(n, mat2, 0, m);
    int** g = getSlice(n, mat2, m, 0);
    int** h = getSlice(n, mat2, m, m);
    CALI_MARK_END(slicing);


    CALI_MARK_BEGIN(adding);
    int** bds = addMatrices(m, b, d, false);
    int** gha = addMatrices(m, g, h, true);
    int** s1 = cudaNaive(m, bds, gha);
    freeMatrix2D(m, bds);
    freeMatrix2D(m, gha);
    

    int** ada = addMatrices(m, a, d, true);
    int** eha = addMatrices(m, e, h, true);
    int** s2 = cudaNaive(m, ada, eha);
    freeMatrix2D(m, ada);
    freeMatrix2D(m, eha);

    int** acs = addMatrices(m, a, c, false);
    int** efa = addMatrices(m, e, f, true);
    int** s3 = cudaNaive(m, acs, efa);
    freeMatrix2D(m, acs);
    freeMatrix2D(m, efa);

    int** aba = addMatrices(m, a, b, true);
    int** s4 = cudaNaive(m, aba, h);
    freeMatrix2D(m, aba);
    freeMatrix2D(m, b);

    int** fhs = addMatrices(m, f, h, false);
    int** s5 = cudaNaive(m, a, fhs);
    freeMatrix2D(m, fhs);
    freeMatrix2D(m, a);
    freeMatrix2D(m, f);
    freeMatrix2D(m, h);

    int** ges = addMatrices(m, g, e, false);
    int** s6 = cudaNaive(m, d, ges);
    freeMatrix2D(m, ges);
    freeMatrix2D(m, g);

    int** cda = addMatrices(m, c, d, true);
    int** s7 = cudaNaive(m, cda, e);
    freeMatrix2D(m, cda);
    freeMatrix2D(m, c);
    freeMatrix2D(m, d);
    freeMatrix2D(m, e);

    int** s1s2a = addMatrices(m, s1, s2, true);
    int** s6s4s = addMatrices(m, s6, s4, false);
    int** c11 = addMatrices(m, s1s2a, s6s4s, true);
    freeMatrix2D(m, s1s2a);
    freeMatrix2D(m, s6s4s);
    freeMatrix2D(m, s1);

    int** c12 = addMatrices(m, s4, s5, true);
    freeMatrix2D(m, s4);

    int** c21 = addMatrices(m, s6, s7, true);
    freeMatrix2D(m, s6);

    int** s2s3s = addMatrices(m, s2, s3, false);
    int** s5s7s = addMatrices(m, s5, s7, false);
    int** c22 = addMatrices(m, s2s3s, s5s7s, true);
    freeMatrix2D(m, s2s3s);
    freeMatrix2D(m, s5s7s);
    freeMatrix2D(m, s2);
    freeMatrix2D(m, s3);
    freeMatrix2D(m, s5);
    freeMatrix2D(m, s7);
    CALI_MARK_END(adding);

    CALI_MARK_BEGIN(merging);
    int** prod = combineMatrices(m, c11, c12, c21, c22);
    CALI_MARK_END(merging);

    freeMatrix2D(m, c11);
    freeMatrix2D(m, c12);
    freeMatrix2D(m, c21);
    freeMatrix2D(m, c22);
    
    CALI_MARK_BEGIN(comp);
    return prod;
}

const char* data_init = "data_init";
const char* comm = "comm";
const char* comp =  "comp";
const char* correctness = "correctness";
const char* strassen_time = "strassen_time";
const char* naive_multi = "naive_multi";

const char* slicing = "slicing";

const char* slicing = "merging";


const char* comm_little = "comm_little";
const char* comp_little = "comp_little"

const char* host2device = "host2device";
const char* device2host = "device2host";
const char* cuda_naive_time = "cuda_naive_time";

int main(int argc, char *argv[])
{
    
    cali::ConfigManager mgr;
    mgr.start();

    int n;
    n = atoi(argv[2]);
    THREADS = atoi(argv[1]);
    
    CALI_MARK_BEGIN(data_init)
    int** mat1 = allocateMatrix2D(n);
    fillMatrix2D(n, mat1);

    int** mat2 = allocateMatrix2D(n);
    fillMatrix2D(n, mat2);
    CALI_MARK_END(data_init)

    clock_t start, end;
    start = clock();

    CALI_MARK_BEGIN(strassen_time)
    int** prod = strassen(n, mat1, mat2);
    CALI_MARK_END(strassen_time)

    end = clock();
    double time = double(end - start) / double(CLOCKS_PER_SEC);
    cout<<"Parallel Strassen Runtime (CUDA): "<<time<<" seconds\n";

    adiak::init(NULL);
    adiak::launchdate();                                         // launch date of the job
    adiak::libraries();                                          // Libraries used
    adiak::cmdline();                                            // Command line used to launch the job
    adiak::clustername();                                        // Name of the cluster
    adiak::value("Algorithm", "CUDA Naive Matrix Multiplication");// The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA");          // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                          // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));              // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", MATRIX_SIZE);                        // The number of elements in input dataset (1000)
    // adiak::value("InputType", inputType);                        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    //adiak::value("num_procs", size);                        // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS);                    // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS);                      // The number of CUDA blocks
    adiak::value("group_num", 8);                     // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online");

    return 0;
}



