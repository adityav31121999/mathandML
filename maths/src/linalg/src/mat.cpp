
#include "include/mat.hpp"

/**
 * @brief Assign mat from mat
 * @param a matrix whose values are assigned to this matrix
 * @return values assigned to matrix
 */
mat mat::operator=(mat a) {
    this->row = a.row;
    this->row = a.col;
    this->a = a.a;
    return mat();
}

/**
 * @brief Assign mat from vector<vector<double>>
 * @param a vector<vector<double>> whose values are assigned to this matrix
 * @return values assigned to matrix
 */
mat mat::operator=(std::vector<std::vector<double>> a) {
    this->row = a.size();
    this->col = a[0].size();
    this->a = a;
    return mat();
}


