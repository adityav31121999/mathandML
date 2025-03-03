
#include "include/vecops.hpp"
#include <stdexcept>
#include <numeric>

/**
 * @brief Calculate the inner product of a vector of vector with itself 
 * and form a square matrix of its dot products.
 * @param a The input matrix
 * @return The product of the matrix with itself
 * @throws std::runtime_error if the input matrix is empty
 */
std::vector<std::vector<double>> iproduct(std::vector<std::vector<double>> a) {
    if(a.empty()) 
        throw std::runtime_error("embeddings must not be empty");

    std::vector<std::vector<double>> c(a.size(), std::vector<double>(a.size(), 0.0));
    
    for(size_t i = 0; i < a.size(); i++) {
        for(size_t j = 0; j < a.size(); j++) {
            c[i][j] = std::inner_product(a[i].begin(), a[i].end(), a[j].begin(), 0.0);
        }
    }
    
    return c;
}

/**
 * @brief Calculate the product of two vector of vectors and form a square 
 * matrix of dot products of each combination of two vectors.
 * @param a The first matrix
 * @param b The second matrix
 * @return The product of the two matrices
 * @throws std::runtime_error if the rows of the matrices are not of equal sizes
 */
std::vector<std::vector<double>> iproduct(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b) {
    if(a.empty() || b.empty() || a[0].size() != b[0].size()) 
        throw std::runtime_error("Rows must be of equal sizes");

    std::vector<std::vector<double>> c(a.size(), std::vector<double>(b[0].size(), 0.0));
    
    for(size_t i = 0; i < a.size(); i++) {
        for(size_t j = 0; j < b.size(); j++) {
            c[i][j] = std::inner_product(a[i].begin(), a[i].end(), b[j].begin(), 0.0);
        }
    }
    
    return c;
}
