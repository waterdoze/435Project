#ifndef LIN_NAIVE_H
#define LIN_NAIVE_H

void cpu_lin_naive(const mat &A, const mat &B, mat &C) {
	size_t A_row = A.size();
	size_t A_column = A[0].size();
	size_t B_row = B.size();
	size_t B_column = B[0].size();
	size_t C_row = C.size();
	size_t C_column = C[0].size();
	
	
    
	for (size_t i = 0; i < A_row; ++i) {
		for (size_t j = 0; j < B_column; ++j) {
			float tmp = 0.0;
			for (size_t k = 0; k < A_column; ++k) {
				tmp += A[i][k] * B[k][j];
			}
			C[i][j] = tmp;
		}
	}
}
#endif