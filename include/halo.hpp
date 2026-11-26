#ifndef HALO_HPP
#define HALO_HPP

#include "partition.hpp"
#include <vector>

// ---------------------------------------------------------------------------------------- //

void halo_exchange(const Partition& part, const std::vector<double>& p, std::vector<double>& p_ext);

// ---------------------------------------------------------------------------------------- //

#endif