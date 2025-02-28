
#ifndef POLY_HPP
#define POLY_HPP 1

#include <iostream>
#include <vector>

template <typename t> class poly {
private:
    std::vector<t> coeffs;
    std::vector<int> exponents;
    int degree;
public:
    poly();
    poly(std::vector<t> coeffs);
    poly(std::vector<t> coeffs, t x);
    poly(std::vector<t> coeffs, t x, t y);
    poly(std::vector<t> coeffs, t x, t y, t z);
}

#endif
