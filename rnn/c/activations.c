
#include "include/activations.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// errors

/**
 * @brief Compute the Mean Squared Error (MSE) of two arrays
 * @param[in] actual The actual values
 * @param[in] expected The expected values
 * @param[in] size The size of the arrays
 * @return The Mean Squared Error
 */
double MSE(const double* actual, const double* expected, size_t size) {
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double diff = actual[i] - expected[i];
        sum += diff * diff;
    }
    return sum / size;
}

/**
 * @brief Compute the Root Mean Squared Error (RMSE) of two arrays
 * @param[in] actual The actual values
 * @param[in] expected The expected values
 * @param[in] size The size of the arrays
 * @return The Root Mean Squared Error
 */
double rMSE(const double* actual, const double* expected, size_t size) {
    return sqrt(MSE(actual, expected, size));
}

// sigmoid functions

/**
 * @brief Sigmoid activation function
 * @param[in] x The input value
 * @return The sigmoid of the input
 */
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/**
 * @brief Derivative of the sigmoid activation function
 * @param[in] x The input value
 * @return The derivative of the sigmoid for the input
 */
double sigmoidder(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// ReLU

/**
 * @brief Rectified Linear Unit (ReLU) activation function
 * @param[in] x The input value
 * @return The ReLU of the input
 */
double ReLU(double x) {
    return (x > 0) ? x : 0;
}

/**
 * @brief Derivative of the ReLU activation function
 * @param[in] x The input value
 * @return The derivative of the ReLU for the input
 */
double ReLUder(double x) {
    return (x > 0) ? 1 : 0;
}

// SELU

/**
 * @brief Scaled Exponential Linear Unit (SeLU) activation function
 * @param[in] x The input value
 * @return The SeLU of the input
 */
double SeLU(double x) {
    return (x > 0) ? x : 0.1 * x;
}

/**
 * @brief Derivative of the SeLU activation function
 * @param[in] x The input value
 * @return The derivative of the SeLU for the input
 */
double SeLUder(double x) {
    return (x > 0) ? 1 : 0.1;
}

// Softmax function

/**
 * @brief Softmax activation function
 * @param[in] input The input values
 * @param[in] size The size of the input
 * @param[in] temperature The temperature of the softmax
 * @return The softmax of the input
 */
double* softmax(const double* input, size_t size, double temperature) {
    double* output = (double*)malloc(size * sizeof(double));
    if (!output) return NULL;

    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        output[i] = exp(input[i] / temperature);
        sum += output[i];
    }
    for (size_t i = 0; i < size; ++i) {
        output[i] /= sum;
    }
    return output;
}

// Softmax derivative

/**
 * @brief Derivative of the softmax activation function
 * @param[in] input The input values
 * @param[in] size The size of the input
 * @param[in] temperature The temperature of the softmax
 * @return The derivative of the softmax for the input
 */
double* softmaxder(const double* input, size_t size, double temperature) {
    double* softmax_vals = softmax(input, size, temperature);
    if (!softmax_vals) return NULL;

    double* output = (double*)malloc(size * sizeof(double));
    if (!output) {
        free(softmax_vals);
        return NULL;
    }

    for (size_t i = 0; i < size; ++i) {
        output[i] = softmax_vals[i] * (1.0 - softmax_vals[i]);
    }
    free(softmax_vals);
    return output;
}

// Softmax for 2D array

/**
 * @brief Softmax activation function for 2D arrays
 * @param[in] input The 2D input
 * @param[in] rows The number of rows in the input
 * @param[in] cols The number of columns in the input
 * @param[in] temperature The temperature of the softmax
 * @return The softmax of the input
 */
double** softmax2D(double** input, size_t rows, size_t cols, double temperature) {
    double** output = (double**)malloc(rows * sizeof(double*));
    if (!output) return NULL;

    for (size_t i = 0; i < rows; ++i) {
        output[i] = softmax(input[i], cols, temperature);
    }
    return output;
}

// Softmax derivative for 2D array

/**
 * @brief Derivative of the softmax activation function for 2D arrays
 * @param[in] input The 2D input
 * @param[in] rows The number of rows in the input
 * @param[in] cols The number of columns in the input
 * @param[in] temperature The temperature of the softmax
 * @return The derivative of the softmax for the input
 */
double** softmaxder2D(double** input, size_t rows, size_t cols, double temperature) {
    double** output = (double**)malloc(rows * sizeof(double*));
    if (!output) return NULL;

    for (size_t i = 0; i < rows; ++i) {
        output[i] = softmaxder(input[i], cols, temperature);
    }
    return output;
}
