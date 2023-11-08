#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <cmath>

#define BLOCK_SIZE 16 // Block size set to 16 for simplicity
#define MAX_DEPTH 20  // Maximum depth of recursion

float *h_A, *h_B, *h_C; // host data
float *d_A[MAX_DEPTH], *d_B[MAX_DEPTH], *d_C[MAX_DEPTH]; // device data for A, B, C
float *d_M1[MAX_DEPTH], *d_M2[MAX_DEPTH], *d_M3[MAX_DEPTH], *d_M4[MAX_DEPTH], *d_M5[MAX_DEPTH], *d_M6[MAX_DEPTH], *d_M7[MAX_DEPTH]; // device data for M1 ~ M7

template <typename T>
__global__ void classicalMatmul(T *A, T *B, T *C, const int dim) // Classical matrix multiplication
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < dim && col < dim)
    {
        T sum = 0;
        for (int k = 0; k < dim; ++k)
        {
            sum += A[row * dim + k] * B[k * dim + col];
        }
        C[row * dim + col] = sum;
    }
}

template <typename T>
void strassenMatmul(cublasHandle_t &handle, T *A, T *B, T *C, const int dim, const int d, const int threshold) // Strassen matrix multiplication
{
    const int dim_2 = dim / 2;

    int lda = dim, ldb = dim, ldc = dim_2;
    int m = dim_2, n = dim_2;
    T one = 1, zero = 0, m_one = -1;

    if (dim <= threshold)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((dim + BLOCK_SIZE - 1) / BLOCK_SIZE, (dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        classicalMatmul<T><<<grid, block>>>(A, B, C, dim);
        // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &one, B, dim, A, dim, &zero, C, dim);
        return;
    }

    /* M1 */
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A, lda, &one, A + dim_2 * dim + dim_2, ldb, d_A[d + 1], ldc);
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B, lda, &one, B + dim_2 * dim + dim_2, ldb, d_B[d + 1], ldc);
    strassenMatmul(handle, d_A[d + 1], d_B[d + 1], d_M1[d + 1], dim_2, d + 1, threshold);

    /* M2 */
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A + dim_2 * dim, lda, &one, A + dim_2 * dim + dim_2, ldb, d_A[d + 1], ldc);
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B, lda, &zero, B, ldb, d_B[d + 1], ldc);
    strassenMatmul(handle, d_A[d + 1], d_B[d + 1], d_M2[d + 1], dim_2, d + 1, threshold);

    /* M3 */
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A, lda, &zero, A, ldb, d_A[d + 1], ldc);
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B + dim_2, lda, &m_one, B + dim_2 * dim + dim_2, ldb, d_B[d + 1], ldc);
    strassenMatmul(handle, d_A[d + 1], d_B[d + 1], d_M3[d + 1], dim_2, d + 1, threshold);

    /* M4 */
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A + dim_2 * dim + dim_2, lda, &zero, A, ldb, d_A[d + 1], ldc);
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B + dim_2 * dim, lda, &m_one, B, ldb, d_B[d + 1], ldc);
    strassenMatmul(handle, d_A[d + 1], d_B[d + 1], d_M4[d + 1], dim_2, d + 1, threshold);

    /* M5 */
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A, lda, &one, A + dim_2, ldb, d_A[d + 1], ldc);
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B + dim_2 * dim + dim_2, lda, &zero, B, ldb, d_B[d + 1], ldc);
    strassenMatmul(handle, d_A[d + 1], d_B[d + 1], d_M5[d + 1], dim_2, d + 1, threshold);

    /* M6 */
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A + dim_2 * dim, lda, &m_one, A, ldb, d_A[d + 1], ldc);
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B, lda, &one, B + dim_2, ldb, d_B[d + 1], ldc);
    strassenMatmul(handle, d_A[d + 1], d_B[d + 1], d_M6[d + 1], dim_2, d + 1, threshold);

    /* M7 */
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A + dim_2, lda, &m_one, A + dim_2 * dim + dim_2, ldb, d_A[d + 1], ldc);
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B + dim_2 * dim, lda, &one, B + dim_2 * dim + dim_2, ldb, d_B[d + 1], ldc);
    strassenMatmul(handle, d_A[d + 1], d_B[d + 1], d_M7[d + 1], dim_2, d + 1, threshold);

    /* C1 */
    lda = dim, ldb = dim / 2, ldc = dim; // C = C + B
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &zero, C, lda, &one, d_M1[d + 1], ldb, C, ldc);
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C, lda, &one, d_M4[d + 1], ldb, C, ldc);
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C, lda, &m_one, d_M5[d + 1], ldb, C, ldc);
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C, lda, &one, d_M7[d + 1], ldb, C, ldc);

    /* C2 */
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &zero, C + dim_2, lda, &one, d_M3[d + 1], ldb, C + dim_2, ldc);
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C + dim_2, lda, &one, d_M5[d + 1], ldb, C + dim_2, ldc);

    /* C3 */
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &zero, C + dim_2 * dim, lda, &one, d_M2[d + 1], ldb, C + dim_2 * dim, ldc);
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C + dim_2 * dim, lda, &one, d_M4[d + 1], ldb, C + dim_2 * dim, ldc);

    /* C4 */
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &zero, C + dim_2 * dim + dim_2, lda, &one, d_M1[d + 1], ldb, C + dim_2 * dim + dim_2, ldc);
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C + dim_2 * dim + dim_2, lda, &m_one, d_M2[d + 1], ldb, C + dim_2 * dim + dim_2, ldc);
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C + dim_2 * dim + dim_2, lda, &one, d_M3[d + 1], ldb, C + dim_2 * dim + dim_2, ldc);
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C + dim_2 * dim + dim_2, lda, &one, d_M6[d + 1], ldb, C + dim_2 * dim + dim_2, ldc);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <dim>\n", argv[0]);
        exit(0);
    }

    /* Initialize */

    int nDim = atoi(argv[1]);
    int threshold = 16; // threshold for switching to classical matrix multiplication
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nDim + BLOCK_SIZE - 1) / BLOCK_SIZE, (nDim + BLOCK_SIZE - 1) / BLOCK_SIZE);

    assert(nDim >= threshold && threshold >= BLOCK_SIZE);

    size_t nBytes = nDim * nDim * sizeof(float);

    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);


    int depth, _dim = nDim;
    for (depth = 0; depth < MAX_DEPTH && _dim > 0; ++depth)
    {
        cudaMalloc((float **)&d_A[depth], _dim * _dim * sizeof(float));
        cudaMalloc((float **)&d_B[depth], _dim * _dim * sizeof(float));

        if (depth == 0)
        {
            cudaMalloc((float **)&d_C[depth], _dim * _dim * sizeof(float));
        }
        else
        {
            cudaMalloc((float **)&d_M1[depth], _dim * _dim * sizeof(float));
            cudaMalloc((float **)&d_M2[depth], _dim * _dim * sizeof(float));
            cudaMalloc((float **)&d_M3[depth], _dim * _dim * sizeof(float));
            cudaMalloc((float **)&d_M4[depth], _dim * _dim * sizeof(float));
            cudaMalloc((float **)&d_M5[depth], _dim * _dim * sizeof(float));
            cudaMalloc((float **)&d_M6[depth], _dim * _dim * sizeof(float));
            cudaMalloc((float **)&d_M7[depth], _dim * _dim * sizeof(float));
        }
        _dim /= 2;
    }

    cudaMemcpy(d_A[0], h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B[0], h_B, nBytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    CudaTimer ct;

    /* Run strassenMatmul */

    ct.start();
    strassenMatmul<float>(handle, d_A[0], d_B[0], d_C[0], nDim, 0, threshold);
    ct.stop();

    cudaMemcpy(h_C, d_C[0], nBytes, cudaMemcpyDeviceToHost);
    printf("[strassenMatmul] %.5fms\n", ct.value() / N_TEST);

    // verify result
    for(int i = 0; i < nDim; ++i)
    {
        for(int j = 0; j < nDim; ++j)
        {
            float sum = 0.0f;
            for(int k = 0; k < nDim; ++k)
            {
                sum += h_A[i * nDim + k] * h_B[k * nDim + j];
            }
            if(fabs(sum - h_C[i * nDim + j]) > 1e-5)
            {
                printf("Error: %f != %f\n", sum, h_C[i * nDim + j]);
                exit(0);
            }
        }
    }
    printf("Verification successful!\n");

    /* Free memory */

    cublasDestroy(handle);

    for (int i = 0; i < depth; ++i)
    {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);

        if (i == 0)
        {
            cudaFree(d_C[i]);
        }
        else
        {
            cudaFree(d_M1[i]);
            cudaFree(d_M2[i]);
            cudaFree(d_M3[i]);
            cudaFree(d_M4[i]);
            cudaFree(d_M5[i]);
            cudaFree(d_M6[i]);
            cudaFree(d_M7[i]);
        }
    }

    cudaDeviceReset();

    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done.\n");

    return 0;
}