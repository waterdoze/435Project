#ifndef CUDA_NAIVE
#define CUDA_NAIVE

__global__ void matrixMultiplication(int* rowsA, int* matrixB, int* rowsC, int n)
{
    int prod = blockIdx.x * blockDim.x + threadIdx.x;
    int i = prod / n;
    int j = prod % n;
    for (int k = 0; k < n; k++) {
        rowC[i * n + j] += mat1[i * n + k] * mat2[k * n + j];
    }
}

void naive(int* matrixA, int* matrixB,int* matrixC,int matSize){
    size_t bytes = n * n * sizeof(int)

    size_t bytes = n * n * sizeof(int);

    int *d_mat1, *d_mat2, *d_product;

    cudaMalloc(&d_mat1, bytes);
    cudaMalloc(&d_mat2, bytes);
    cudaMalloc(&d_product, bytes);

    cudaMemcpy(d_mat1, h_mat1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, h_mat2, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_product, h_product, bytes, cudaMemcpyHostToDevice);

    int* rowsA[matSize/threads][matSize];
    int* rowsC[matSize/threads][matSize];

    dim3 blocks(BLOCKS,1);
    dim3 threads(THREADS,1);

}

#endif