#ifndef CSR_HPP
#define CSR_HPP

#include "partition.hpp"
#include <vector>

struct CSR {
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<int> values;
    int num_rows;
};

// ---------------------------------------------------------------------------------------- //

// Return CSR matrix
CSR build_matrix(Partition part);

// Outputs CSR matrix
void output_matrix(CSR matrix);

// ---------------------------------------------------------------------------------------- //

#endif