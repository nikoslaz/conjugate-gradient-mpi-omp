#include "vec_ops.hpp"

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum    = 0;
    double result = 0;
    const int len = static_cast<int>(a.size());
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }

    MPI_Allreduce(&sum, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return result;
}

void axpy(double alpha, const std::vector<double>& x, std::vector<double>& y) {
    const int len = static_cast<int>(x.size());
    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
        y[i] = y[i] + alpha * x[i];
    }
}