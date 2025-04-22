
// activations.hpp: header source for activations
#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP 1

#include <vector>

// errors

double MSE(std::vector<double>, std::vector<double>);
double rMSE(std::vector<double>, std::vector<double>);

// activations

double sigmoid(double);
double sigmoidder(double);
double ReLU(double);
double ReLUder(double);
double SeLU(double);
double SeLUder(double);
std::vector<double> softmax(std::vector<double>, double);
std::vector<double> softmaxder(std::vector<double>, double);
std::vector<std::vector<double>> softmax(std::vector<std::vector<double>>, double);
std::vector<std::vector<double>> softmaxder(std::vector<std::vector<double>>, double);

#endif
