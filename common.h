#include <iostream>
#include <random>
#include <vector>

void get_random_matrix(std::vector<std::vector<float>> &matrix, const float &min, const float &max) {
    std::random_device rd;
    std::default_random_engine engine{rd()};
    std::uniform_real_distribution<float> uniform(min, max);
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            matrix[i][j] = uniform(engine);
        }
    }
}