#include <iostream>
#include <cstdlib>
#include <vector>

void get_random_matrix(std::vector<std::vector<int>> &matrix) {
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            matrix[i][j] = rand() % (10) + 1; // random number between 1 and 10
        }
    }
}

void print_matrix(const std::vector<std::vector<int>> matrix) {
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}