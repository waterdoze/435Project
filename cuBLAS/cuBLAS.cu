#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>


int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <matrix_dimension>\n", argv[0]);
        return 1;
    }

    // Matrix dimension
    int n = atoi(argv[1);

    // Allocate memory for matrices on the host
    float *h_A = (float *)malloc(n * n * sizeof(float));
    float *h_B = (float *)malloc(n * n * sizeof(float));
    float *h_C = (float *)malloc(n * n * sizeof(float));

    // Initialize matrices A and B with random values
    for(int i = 0; i < n * n; i++)
    {
        h_A[i] = rand() % 10 + 1;
        h_B[i] = rand() % 10 + 1;
    }

    // Allocate memory for matrices on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, n * n * sizeof(float));
    cudaMalloc((void **)&d_B, n * n * sizeof(float));
    cudaMalloc((void **)&d_C, n * n * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix multiplication using cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n);

    // Copy the result matrix from device to host
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize(); // Wait for the GPU to finish

    // verify the result using naive method
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            float sum = 0.0f;
            for(int k = 0; k < n; k++)
            {
                sum += h_A[i * n + k] * h_B[k * n + j];
            }
            if(fabs(h_C[i * n + j] - sum) > 1e-5)
            {
                printf("Error: %f != %f\n", h_C[i * n + j], sum);
                return 1;
            }
        }
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
