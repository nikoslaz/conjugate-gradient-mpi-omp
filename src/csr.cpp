#include "csr.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

CSR build_matrix(Partition part) {
    CSR csr;
    csr.num_rows = part.local_n;
    csr.col_idx.reserve(part.local_n * 5);
    csr.row_ptr.reserve(part.local_n + 1);

    csr.row_ptr.push_back(0);
    int start = part.first_grid_row;
    int end   = part.first_grid_row + part.local_grid_rows;
    for (int r = start; r < end; r++) {
        for (int c = 0; c < part.n; c++) {
            int i = r * part.n + c;
            int li = i - part.global_offset + part.n;

            if (r > 0) { // above me
                csr.col_idx.push_back(li - part.n);
                csr.values.push_back(-1);
            }
            if (c > 0) { // left 
                csr.col_idx.push_back(li - 1);
                csr.values.push_back(-1);
            }

            // diagonal
            csr.col_idx.push_back(li);
            csr.values.push_back(4);
            
            if (c < part.n - 1) { // right
                csr.col_idx.push_back(li + 1);
                csr.values.push_back(-1);
            }
            if (r < part.n - 1) { // below me
                csr.col_idx.push_back(li + part.n);
                csr.values.push_back(-1);
            }

            csr.row_ptr.push_back(csr.values.size()); // marks where the next row starts
        }       
    }

    return csr;
}

void output_matrix(CSR matrix) {
    const int n = matrix.num_rows;

    // Local columns include the halos, so the dense view can be wider
    // than num_rows: use the largest column index + 1 as the width.
    int cols = 0;
    for (int col : matrix.col_idx) {
        cols = std::max(cols, col + 1);
    }

    // Expand the 1D CSR arrays into a dense n x cols grid.
    std::vector<std::vector<int>> dense(n, std::vector<int>(cols, 0));
    for (int r = 0; r < n; r++) {
        for (int k = matrix.row_ptr[r]; k < matrix.row_ptr[r + 1]; k++) {
            dense[r][matrix.col_idx[k]] = matrix.values[k];
        }
    }

    int width = 1;
    for (const auto& row : dense) {
        for (int v : row) {
            width = std::max(width, static_cast<int>(std::to_string(v).size()));
        }
    }

    for (const auto& row : dense) {
        for (int v : row) {
            std::cout << std::setw(width + 1) << v;
        }
        std::cout << '\n';
    }
    std::cout << std::flush;
}