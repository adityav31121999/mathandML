
#include "include/activations.hpp"
#include <functional>
#include <algorithm>
#include <numeric>

//----------------SIGMOID----------------//

/**
 * @brief Derivative of sigmoid activation function. Calculates the derivative of the sigmoid function.
 *      The derivative of sigmoid(x) is given by: sigmoid_derivative(x) = sigmoid(x) * (1 - sigmoid(x))
 * @param x Input value
 * @return The derivative of sigmoid(x)
 */
double sigmoidder(double x) {
    // Calculate the sigmoid of x
    double s = sigmoid(x);
    // Calculate the derivative of sigmoid(x)
    return s * (1 - s);
}

/**
 * @brief Derivative of sigmoid activation function. This function calculates 
 *      the derivative of the sigmoid function for each element of the input vector.
 * @param x Input vector
 * @return A vector where each element is the derivative of the corresponding 
 *      element in the input vector.
 */
std::vector<double> sigmoidvder(std::vector<double> x) {
    // Create a copy of the input vector
    std::vector<double> y(x);
    // Apply the derivative of sigmoid function to each element of the input vector
    std::transform(x.begin(), x.end(), y.begin(), [](double& i){ return sigmoidder(i); });
    return y;
}

/**
 * @brief Derivative of sigmoid activation function. This function calculates 
 *      the derivative of the sigmoid function for each element of the input matrix.
 * @param x Input matrix
 * @return A matrix where each element is the derivative of the corresponding 
 *      element in the input matrix.
 */
std::vector<std::vector<double>> sigmoidder(std::vector<std::vector<double>> x) {
    std::vector<std::vector<double>> result(x.size(), std::vector<double>(x[0].size()));
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sigmoidvder(x[i]);
    }
    return result;
}

//----------------SOFTMAX----------------//

/**
 * @brief Derivative of softmax activation function This function calculates the derivative of 
 *          the softmax function for a vector of input values. The derivative of the softmax 
 *          function is given by: softmax_derivative(x) = softmax(x) * (1 - softmax(x)). The 
 *          derivative of softmax(x) is a vector of the same size as the input vector, where each 
 *          element is the derivative of the softmax function with respect to the corresponding 
 *          input value.
 * @param x Input vector
 * @param temp Temperature parameter for the softmax function. A high temperature results in a more 
 *          uniform distribution.
 * @return The derivative of softmax(x) for each input value
 */
std::vector<double> softmaxder(std::vector<double> x, double temp = 1.0) {
    // Create a copy of the input vector
    std::vector<double> y(x);
    // Calculate the sum of exponential of each element of the input vector
    double sum = 0;
    std::for_each(y.begin(), y.end(), [&sum, temp](double& val) { sum += exp(val/temp); });
    // Calculate the softmax of each element of the input vector
    std::for_each(y.begin(), y.end(), [&sum](double& val) { val = exp(val) / sum; });
    // Calculate the derivative of softmax(x) for each input value
    std::vector<double> result(y.size(), 0.0);
    for (size_t i = 0; i < y.size(); ++i) {
        // Calculate the derivative of softmax(x) using the formula: softmax_derivative(x) = softmax(x) * (1 - softmax(x))
        result[i] = y[i] * (1 - y[i]);
        // Subtract the softmax of each other element from the derivative of softmax(x)
        for (size_t j = 0; j < y.size(); ++j) {
            if (i == j) {
                continue;
            }
            result[i] -= y[j];
        }
    }
    // Return the vector of derivatives
    return result;
}

/**
 * @brief Derivative of softmax activation function. Calculates the derivative of the softmax function for each element of a 2D vector.
 *          The derivative of the softmax function is given by: softmax_derivative(x) = softmax(x) * (1 - softmax(x)). The
 *          derivative of softmax(x) is a vector of the same size as the input vector, where each
 *          element is the derivative of the softmax function with respect to the corresponding
 *          input value.
 * @param x Input 2D vector
 * @param temp Temperature parameter for the softmax function. A high temperature results in a more
 *          uniform distribution.
 * @return The derivative of softmax(x) for each input value
 */
std::vector<std::vector<double>> softmaxder(std::vector<std::vector<double>> x, double temp = 1.0) {
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
    std::vector<std::vector<double>> result(y.size(), std::vector<double>(y[0].size()));       // Initialize the result vector
    // Calculate the derivative of softmax(x) for each input value
    for (size_t i = 0; i < y.size(); ++i) {
        for (size_t j = 0; j < y[0].size(); ++j) {
            result[i][j] = y[i][j] * (1 - y[i][j]);
            // Subtract the softmax of each other element from the derivative of softmax(x)
            std::transform(y[i].begin(), y[i].end(), result[i].begin(), [i, &y](double val){ 
                double sum = 0.0;
                for (size_t k = 0; k < y[0].size(); ++k) {
                    if (k == i) {
                        continue;
                    }
                    sum += y[i][k];
                }
                return val * (1 - val) - sum;
            });
        }
    }
    return y;
}

//----------------ReLU----------------//

/**
 * @brief Calculates the derivative of the ReLU (Rectified Linear Unit)
 *      activation function. The derivative of the ReLU function is 0 if the input value is less than 0,
 *      and 1 otherwise.
 * @param x Input value
 * @return 0 if x < 0, 1 otherwise
 */
double ReLUder(double x) {
    // The derivative of the ReLU function is 0 if the input value is less than 0, and 1 otherwise.
    return (x > 0) ? 1 : 0; // Return 1 if x > 0, 0 otherwise
}

/**
 * @brief Calculate the derivative of the ReLU activation function for each element in a vector.
 *      The derivative of the ReLU function is 0 if the input value is less than 0, and 1 otherwise.
 *      This function applies the derivative of the ReLU function to each element in the input vector.
 * @tparam t type of the elements in the input vector
 * @param x input vector
 */
std::vector<double> ReLUvder(std::vector<double> x) {
    std::vector<double> y(x);
    // Use std::transform to apply ReLU_derivative to each element in x
    std::transform(x.begin(), x.end(), y.begin(), [](double& i){ return ReLUder(i); });
    return y;
}

//----------------SeLU----------------//

/**
 * @brief The derivative of the SeLU (Scale-Exponential Linear Unit) activation function.
 *        The SeLU derivative is defined as:
 *        f'(x) = 1 if x > 0
 *        f'(x) = 0.1 otherwise
 * @param x input value
 * @return the value of the SeLU derivative function applied to the input
 */
double SeLUder(double x) {
    // Apply the SeLU derivative function to the input
    return (x > 0) ? 1 : 0.1;
}

/**
 * @brief Calculates the derivative of the SeLU (Scale-Exponential Linear Unit) activation function 
 * for each element in a vector. 
 * The SeLU derivative is defined as:
 *        f'(x) = 1 if x > 0
 *        f'(x) = 0.1 otherwise
 * @param x Input vector
 * @return A vector where each element is the derivative of the SeLU function for the corresponding element in the input vector.
 */
std::vector<double> SeLUvder(std::vector<double> x) {
    // Create a copy of the input vector
    std::vector<double> y(x);
    // Apply the SeLU derivative function to each element of the input vector
    std::transform(x.begin(), x.end(), y.begin(), [](double& i){ return SeLUder(i); });
    return y;
}

//----------------Least of them all (LOTA----------------//

/**
 * @brief Calculates the derivative of the LOTA (Least Of Them All) activation function for a vector.
 *        This function calculates the derivative of the LOTA function for each element in a vector.
 *        The LOTA derivative is defined as:
 *        f'(x) = (sum - x) / sum^2 for normalization
 * @param y Input vector
 * @return A vector where each element is the derivative of the LOTA function applied to the corresponding element in the input vector.
 */
std::vector<double> LOTAder(std::vector<double>& y) {
    // Create a copy of the input vector
    std::vector<double> v(y);
    // Find the minimum value in the entire vector
    double min_val =  *std::min_element(v.begin(), v.end());
    min_val = std::abs(min_val);
    // Subtract the minimum value from each element in the vector
    std::transform(v.begin(), v.end(), v.begin(), [&min_val](double& i){ return (i + min_val); });
    // Calculate the sum of the elements in the vector
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    // Normalize the vector by dividing each element by the sum
    std::transform(v.begin(), v.end(), v.begin(), [&sum](double& i){ return ((sum - i) / std::pow(sum, 2)); });
    return v;
}

/**
 * @brief Derivative of the LOTA (Least Of Them All) activation function for a 2D vector.
 *        This function calculates the derivative of the LOTA function for each element
 *        in a 2D vector. The LOTA derivative is defined as:
 *        f'(x) = (sum - x) / sum^2 for normalization
 * @param y Input 2D vector
 * @return A 2D vector where each element is the derivative of the LOTA function applied 
 *         to the corresponding element in the input vector.
 */
std::vector<std::vector<double>> LOTAder(std::vector<std::vector<double>> y) {
    // Create a copy of the input 2D vector
    std::vector<std::vector<double>> x(y);
    // Find the minimum value in the entire 2D vector
    double min_val = 0.0; 
    for (const auto& v: x) {
        double val = *std::min_element(v.begin(), v.end());
        if (val < min_val) {
            min_val = val;
        }
    }
    min_val = std::abs(min_val);
    // Subtract the minimum value from each element in the 2D vector
    for (auto& v : x) {
        std::transform(v.begin(), v.end(), v.begin(), [&min_val](double& i) {
            return (i + min_val);
        });
    }
    double sum = 0.0; // Variable to store the sum of all elements
    // Calculate the sum of all elements in the 2D vector
    for (const auto& v: x) {
        sum += std::accumulate(v.begin(), v.end(), 0.0);
    }
    // Calculate the derivative of the LOTA function for each element
    for (auto& v : x) {
        std::transform(v.begin(), v.end(), v.begin(), [&sum](double& i) {
            return ((sum - i) / std::pow(sum, 2));
        });
    }
    return x; // Return the derived 2D vector
}
