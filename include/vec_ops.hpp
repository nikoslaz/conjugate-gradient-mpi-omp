#ifndef VEC_OPS_HPP
#define VEC_OPS_HPP

#include "spmv.hpp"
#include <mpi.h>

// ---------------------------------------------------------------------------------------- //

// the dot product, giving one number.
double dot(const std::vector<double>& a, const std::vector<double>& b);

// a vector update, y = y + alpha·x, done entry by entry
void axpy(double alpha, const std::vector<double>& x, std::vector<double>& y);

// ---------------------------------------------------------------------------------------- //

#endif