
#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <stdlib.h>
#include <math.h>

// Error functions
double MSE(const double* actual, const double* expected, size_t size);
double rMSE(const double* actual, const double* expected, size_t size);

// Activation functions
double sigmoid(double x);
double sigmoidder(double x);
double ReLU(double x);
double ReLUder(double x);
double SeLU(double x);
double SeLUder(double x);

double* softmax(const double* input, size_t size, double temperature);
double* softmaxder(const double* input, size_t size, double temperature);

double** softmax2D(double** input, size_t rows, size_t cols, double temperature);
double** softmaxder2D(double** input, size_t rows, size_t cols, double temperature);

#endif // ACTIVATIONS_H
