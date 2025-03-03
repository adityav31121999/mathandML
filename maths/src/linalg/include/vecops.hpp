
#ifndef VECOPS_HPP
#define VECOPS_HPP 1

#include <vector>

// vec1.cpp

std::vector<double> operator+(std::vector<double>, std::vector<double>);
std::vector<double> operator-(std::vector<double>, std::vector<double>);
std::vector<double> operator*(std::vector<double>, double);
std::vector<double> operator/(std::vector<double>, double);
std::vector<std::vector<double>> operator+(std::vector<std::vector<double>>, std::vector<std::vector<double>>);
std::vector<std::vector<double>> operator-(std::vector<std::vector<double>>, std::vector<std::vector<double>>);
std::vector<std::vector<double>> operator*(std::vector<std::vector<double>>, double y);
std::vector<std::vector<double>> operator/(std::vector<std::vector<double>>, double y);
bool operator==(std::vector<double>, std::vector<double>);
bool operator!=(std::vector<double>, std::vector<double>);
double sum(std::vector<double>);
double sum(std::vector<std::vector<double>>);
double product(std::vector<double>);
double product(std::vector<std::vector<double>>);

// vec2.cpp

double errorofv(std::vector<double> , std::vector<double> );
double gradientdesc1(std::vector<double>, std::vector<double>);
double vdotv2val(std::vector<double>, std::vector<double>);
double vdotv2scal(std::vector<double> , std::vector<double>);

std::vector<double> error(std::vector<double>, std::vector<double>);
std::vector<double> percenterrorofvec(std::vector<double> , std::vector<double>);
std::vector<double> gradient_descent(std::vector<double>, std::vector<double>, double);

// vec3.cpp

std::vector<double> sumofrow(std::vector<std::vector<double>>);
std::vector<double> sumofcol(std::vector<std::vector<double>>);
std::vector<std::vector<double>> vxv2mat(std::vector<double>, std::vector<double>);
std::vector<double> vxv2v(std::vector<double>, std::vector<double>);
std::vector<double> vdotv2v(std::vector<double>, std::vector<double>);
std::vector<std::vector<double>> vdotmat2mat(std::vector<double>, std::vector<std::vector<double>>);
std::vector<double> vxmat2vec(std::vector<double>, std::vector<std::vector<double>>);
std::vector<std::vector<double>> kronecker(std::vector<std::vector<double>>, std::vector<std::vector<double>>);
std::vector<std::vector<double>> kronecker(std::vector<std::vector<double>>, std::vector<double>);
std::vector<std::vector<double>> hadamard(std::vector<std::vector<double>>, std::vector<std::vector<double>>);
std::vector<std::vector<double>> vec2mat(std::vector<double>, unsigned int, unsigned int);
std::vector<double> mat2vec(std::vector<std::vector<double>>);
std::vector<double> abs(std::vector<double>);
std::vector<double> sqrt(std::vector<double>);
std::vector<double> log10(std::vector<double>);
std::vector<double> loge(std::vector<double>);
std::vector<double> loga(std::vector<double>, int base_a);
std::vector<double> power(std::vector<double>, double);
std::vector<std::vector<double>> power(std::vector<std::vector<double>>, double);





#endif
