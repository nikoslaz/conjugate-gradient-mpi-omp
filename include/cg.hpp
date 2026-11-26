#ifndef CG_HPP
#define CG_HPP

#include "csr.hpp"

// ---------------------------------------------------------------------------------------- //

// Conjugate Gradient algorithm
int cg(const CSR& A, const Partition& part, const std::vector<double>& b, std::vector<double>& x, int max_iter, double tol);

// ---------------------------------------------------------------------------------------- //

#endif