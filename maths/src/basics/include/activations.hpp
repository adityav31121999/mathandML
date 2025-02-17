
#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP 1

#include <vector>

double sigmoid(double);
double sigmoidder(double);
std::vector<double> sigmoidv(std::vector<double>);
std::vector<double> sigmoidvder(std::vector<double>);
std::vector<std::vector<double>> sigmoid(std::vector<std::vector<double>>);
std::vector<std::vector<double>> sigmoidder(std::vector<std::vector<double>>);
std::vector<double> softmax(std::vector<double>, double);
std::vector<double> softmaxder(std::vector<double>, double);
std::vector<std::vector<double>> softmax(std::vector<std::vector<double>>, double);
std::vector<std::vector<double>> softmaxder(std::vector<std::vector<double>>, double);
double ReLU(double);
double ReLUder(double);
std::vector<double> ReLUv(std::vector<double>);
std::vector<double> ReLUvder(std::vector<double>);
double SeLU(double);
double SeLUder(double);
std::vector<double> SeLUv(std::vector<double>);
std::vector<double> SeLUvder(std::vector<double>);
std::vector<double> LOTA(std::vector<double> y);
std::vector<std::vector<double>> LOTA(std::vector<std::vector<double>>);
std::vector<double> LOTAder(std::vector<double> y);
std::vector<std::vector<double>> LOTAder(std::vector<std::vector<double>>);

#endif
