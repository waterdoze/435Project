#ifndef CUDA_NAIVE
#define CUDA_NAIVE

__global__ void matrixMultiplication(int* mat1, int* mat2, int* product, int n)
{
    int prod = blockIdx.x * blockDim.x + threadIdx.x;
    int i = prod / n;
    int j = prod % n;
    for (int k = 0; k < n; k++) {
        product[i * n + j] += mat1[i * n + k] * mat2[k * n + j];
    }
}

#endif