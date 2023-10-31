#include <iostream>
#include <cstdlib>
#include <vector>

#define mat std::vector<std::vector<int>>

void get_random_matrix(mat &matrix) {
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            matrix[i][j] = rand() % 10 + 1; // random number between 1 and 10
        }
    }
}

void print_matrix(const mat matrix) {
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

bool verify(mat m1, mat m2) {
    if(m1.size() != m2.size() || m1[0].size() != m2[0].size()) {
        return false;
    }

    for(int i = 0; i < m1.size(); i++) {
        for(int j = 0; j < m1[0].size(); j++) {
            if(m1[i][j] != m2[i][j]) {
                return false;
            }
        }
    }
    return true;
}

mat addsub_matricies(int n, mat m1, mat m2, bool add) {
    mat ret(n, std::vector<int>(n));

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(add) {
                result[i][j] = m1[i][j] + m2[i][j];
            }
            else {
                result[i][j] = m1[i][j] - m2[i][j];
            }
        }
    }
    return ret;
}

mat combine_matricies(int m, mat m11, mat m12,
        mat m21, mat m22) {

    int n = m * 2;
    mat ret(n, std::vector<int>(n));

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            ret[i][j] = m11[i][j];
            ret[i][j + n] = m12[i][j];
            ret[i + n][j] = m21[i][j];
            ret[i + n][j + n] = m22[i][j];
        }
    }
    return ret;
}


mat split(int n, mat m, int offsetx, int offsety) {
    int new_size = n / 2;
    mat ret(new_size, std::vector<int>(new_size));

    for(int i = 0; i < new_size; i++) {
        for(int j = 0; j < new_size; j++) {
            ret[i][j] = m[i + offsetx][j + offsety];
        }
    }
    return ret;
}