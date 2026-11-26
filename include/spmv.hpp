#ifndef SPMV_HPP
#define SPMV_HPP

#include "csr.hpp"

// ---------------------------------------------------------------------------------------- //

// Computes y = A * x
void spmv(const CSR& A, const std::vector<double>& x, std::vector<double>& y);

// Output spmv
void output_spmv(std::vector<double>& y);

// ---------------------------------------------------------------------------------------- //

#endif