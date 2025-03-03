
#include "include/activations.hpp"
#include <functional>
#include <algorithm>
#include <numeric>

//----------------SIGMOID----------------//

/**
 * @brief Sigmoid activation function. Applies the sigmoid function to the input value.
 * @param x Input value
 * @return Sigmoid of x
 */
double sigmoid(double x) {
    // The sigmoid function is defined as 1 / (1 + exp(-x)).
    return (1 / (1 + std::exp(-x)));
}

/**
 * @brief Applies the sigmoid function to each element of a vector.
 * @param x A reference to the input vector. The function modifies this
 * vector in-place.
 * @return Sigmoid of vector: element-wise
 */
std::vector<double> sigmoidv(std::vector<double> x) {
    std::vector<double> y(x);
    std::transform(x.begin(), x.end(), y.begin(), [](double& i){ return sigmoid(i); });
    return y;
}

/**
 * @brief Apply sigmoid function to each element of a matrix.
 * @param x A reference to the input matrix. 
 * @return Sigmoid of matrix: element-wise
 */
std::vector<std::vector<double>> sigmoid(std::vector<std::vector<double>> x) {
    std::vector<std::vector<double>> result(x.size(), std::vector<double>(x[0].size()));
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sigmoidv(x[i]);
    }
    return result;
}

//----------------SOFTMAX----------------//

/**
 * @brief Softmax activation function. Applies the softmax function to each element of a vector.
 *          The softmax function is used to map a vector of real numbers to a probability distribution.
 *          It is often used in the output layer of a neural network to output probabilities.
 *          The softmax function is defined as follows: softmax(x) = exp(x) / sum(exp(x))
 * @param x The input vector
 * @param temp The temperature parameter for the softmax function. A high temperature results in a more 
 *          uniform distribution.
 * @return Vector of softmax values
 */
std::vector<double> softmax(std::vector<double> x, double temp = 1.0) {
    // Create a copy of the input vector
    std::vector<double> y(x);
    // Calculate the sum of the exponentials of the vector elements
    double sum = 0;
    // Calculate the sum of the exponentials of the vector elements
    std::transform(y.begin(), y.end(), y.begin(), [temp](double& val) { return exp(val / temp); });
    sum = std::accumulate(y.begin(), y.end(), 0.0);
    // Normalize the exponentials by dividing each by the sum
    std::transform(y.begin(), y.end(), y.begin(), [sum](double val) {
        return val / sum;
    });
    // Return the normalized vector
    return y;
}

/**
 * @brief Softmax activation function. Applies the softmax function to each element of a 2D vector.
 *          The softmax function is used to map a vector of real numbers to a probability distribution.
 *          It is often used in the output layer of a neural network to output probabilities.
 *          The softmax function is defined as follows: softmax(x) = exp(x) / sum(exp(x))
 * @param x The 2D input vector
 * @param temp The temperature parameter for the softmax function. A high temperature results in a more 
 *          uniform distribution.
 * @return Vector of softmax values
 */
std::vector<std::vector<double>> softmax(std::vector<std::vector<double>> x, double temp = 1.0) {
    // Create a copy of the input vector
    std::vector<std::vector<double>> y(x);
    // Calculate the sum of the exponentials of the vector elements
    double sum = 0.0;
    for (auto& v : x) {
        std::transform(v.begin(), v.end(), v.begin(), [&temp](double& i){ return exp(i/temp); });
        sum += std::accumulate(v.begin(), v.end(), 0.0);
    }
    // Normalize each element by dividing it by the total sum
    for (auto& v: x) {
        std::transform(v.begin(), v.end(), v.begin(), [&sum](double& i){ return i / sum; });
    }
    return y;
}

//----------------ReLU----------------//

/**
 * @brief ReLU activation function. Calculates the ReLU of a single input value.
 * @param x Input value
 * @return The ReLU of x, which is the maximum of 0 and x
 */
double ReLU(double x) {
    // The ReLU function is defined as max(0, x)
    return std::max(double(0), x); // Return the maximum of 0 and x
}

/**
 * @brief ReLU activation function. Applies the ReLU function to each element of a vector.
 * @param x Input vector
 * @return A vector where each element is the ReLU of the corresponding element in the input vector.
 */
 std::vector<double> ReLUv(std::vector<double> x) {
    // Create a copy of the input vector
    std::vector<double> y(x);
    // Apply the ReLU function to each element of the input vector
    std::transform(x.begin(), x.end(), y.begin(), [](double& i){ return ReLU(i); });
    return y;
}

//----------------SeLU----------------//

/**
 * @brief Applies the SeLU (Scale-Exponential Linear Unit) activation function to the input.
 *      The SeLU function is defined as:
 *      f(x) = x if x > 0
 *      f(x) = 0.1 * x otherwise
 * @param x input value
 * @return the value of the SeLU function applied to the input
 */
double SeLU(double x) {
    // Apply the SeLU function to the input
    return (x > 0) ? x : 0.1 * x;
}

/**
 * @brief Applies the SeLU (Scale-Exponential Linear Unit) activation function to each element in a vector.
 * The SeLU function is defined as:
 *      f(x) = x if x > 0
 *      f(x) = 0.1 * x otherwise
 * @param x Input vector
 * @return A vector where each element is the SeLU of the corresponding element in the input vector.
 */
std::vector<double> SeLUv(std::vector<double> x) {
    std::vector<double> y(x);
    // Use std::transform to apply SeLU to each element in x
    std::transform(x.begin(), x.end(), y.begin(), [](double& i){ return SeLU(i); });
    return y;
}

//----------------Least of them all----------------//

/**
 * @brief Applies the LOTA (Least Of Them All) activation function to a 2D vector.
 *        The LOTA function is defined as:
 *        f(x) = x - min(x) for each element, and
 *        f(x) = f(x) / sum(f(x)) for normalization
 * @param y Input 2D vector
 * @return A 2D vector where each vector is the result of the LOTA function applied to the corresponding vector in the input.
 */
std::vector<double> LOTA(std::vector<double>& y) {
    // Create a copy of the input vector
    std::vector<double> x(y);
    // Find the minimum value in the input vector
    double min_val = 0.0;
    min_val = *std::min_element(x.begin(), x.end());
    min_val = std::abs(min_val);
    // Subtract the minimum value from each element in the vector
    std::transform(x.begin(), x.end(), x.begin(), [&min_val](double& i){ return (i + min_val); });
    // Calculate the sum of the elements in the vector
    double sum = std::accumulate(x.begin(), x.end(), 0.0);
    // Normalize the vector by dividing each element by the sum
    std::transform(x.begin(), x.end(), x.begin(), [&sum](double& i){ return i / sum; });
    return x;
}

/**
 * @brief Applies the LOTA (Least Of Them All) activation function to a 2D vector.
 *        The LOTA function is defined as:
 *        f(x) = x - min(x) for each element, and
 *        f(x) = f(x) / sum(f(x)) for normalization
 * @param y Input 2D vector
 * @return A 2D vector where each vector is the result of the LOTA function applied to the corresponding vector in the input.
 */
std::vector<std::vector<double>> LOTA(std::vector<std::vector<double>> y) {
    // Create a copy of the input 2D vector
    std::vector<std::vector<double>> x(y);
    // Find the minimum value in the entire 2D vector
    double min_val = 0.0;
    for (const auto& v: x) {
        double val = *std::min_element(v.begin(), v.end());
        if (val < min_val) { min_val = val; }
    }
    min_val = std::abs(min_val);
    // Subtract the minimum value from each element in the 2D vector
    for (auto& v : x) {
        std::transform(v.begin(), v.end(), v.begin(), [&min_val](double& i){ return (i + min_val); });
    }
    double sum = 0.0; // Variable to store the sum of all elements
    // Calculate the sum of all elements in the 2D vector
    for (const auto& v: x) {
        sum += std::accumulate(v.begin(), v.end(), 0.0);
    }
    // Normalize each element by dividing it by the total sum
    for (auto& v: x) {
        std::transform(v.begin(), v.end(), v.begin(), [&sum](double& i){ return i / sum; });
    }
    return x; // Return the normalized 2D vector
}

/**
 * @brief Applies the LOTA (Least Of Them All) activation function to a 2D vector.
 *        The LOTA function is defined as:
 *        f(x) = x - min(x) for each element, and
 *        f(x) = f(x) / sum(f(x)) for normalization
 * @param y Input 2D vector
 * @return A 2D vector where each vector is the result of the LOTA function applied to the corresponding vector in the input.
 */
std::vector<std::vector<double>> LOTA(std::vector<std::vector<double>> y, int t) {
    if(y.size() == t) {
        std::vector<std::vector<double>> x(y);
        x = LOTA(x);
        return x;
    }
    // Create a copy of the input 2D vector
    std::vector<std::vector<double>> x(y);
    // Find the minimum value in the entire 2D vector
    double min_val = 0.0;
    for (int i = 0; i < t; i++) {
        for(int j = 0; j < t; j++) {
            if (x[i][j] < min_val)
                min_val = x[i][j];
        }
    }
    min_val = std::abs(min_val);
    // Subtract the minimum value from each element in the 2D vector
    for (int i = 0; i < t; i++) {
        std::transform(x[i].begin(), x[i].begin() + t, x[i].begin(), [&min_val](double& i){ return (i + min_val); });
    }
    double sum = 0.0; // Variable to store the sum of all elements
    // Calculate the sum of all elements in the 2D vector
    for (int i = 0; i < t; i++) {
        sum += std::accumulate(x[i].begin(), x[i].begin() + t, 0.0);
    }
    // Normalize each element by dividing it by the total sum
    for (int i = 0; i < t; i++) {
        std::transform(x[i].begin(), x[i].begin() + t, x[i].begin(), [&sum](double& i){ return i / sum; });
    }
    return x; // Return the normalized 2D vector
}
