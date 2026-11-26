#include "spmv.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

void spmv(const CSR& A, const std::vector<double>& x, std::vector<double>& y) {
    #pragma omp parallel for
    for (int i = 0; i < A.num_rows; i++){
        
        double sum = 0;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i+1]; k++) {
            sum += A.values[k] * x[A.col_idx[k]];
        }

        y[i] = sum;
    }
}

void output_spmv(std::vector<double>& y) {
    const int n = static_cast<int>(std::lround(std::sqrt(static_cast<double>(y.size()))));

    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            std::cout << std::setw(4) << y[r * n + c];
        }
        std::cout << '\n';
    }
    std::cout << std::flush;
}