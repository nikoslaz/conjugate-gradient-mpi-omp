#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "partition.hpp"
#include "csr.hpp"
#include "spmv.hpp"
#include "vec_ops.hpp"
#include "halo.hpp"
#include "cg.hpp"

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = (argc > 1) ? std::atoi(argv[1]) : 10;

    if (size > n) {
        if (rank == 0) std::cerr << "Error: need ranks <= n\n";
        MPI_Finalize();
        return 1;
    }

    Partition part = fill_part(n, rank, size);
    CSR A = build_matrix(part);

    // 1. known solution: all 1s (my own piece)
    std::vector<double> x_true(part.local_n, 1.0);

    // 2. b = A * x_true  (halo + spmv, like the test before)
    std::vector<double> x_true_ext;
    std::vector<double> b(part.local_n, 0.0);
    halo_exchange(part, x_true, x_true_ext);
    spmv(A, x_true_ext, b);

    // 3. solve A x = b, and time it
    std::vector<double> x;
    MPI_Barrier(MPI_COMM_WORLD);                 // everyone starts together
    double t0 = MPI_Wtime();
    int iters = cg(A, part, b, x, 2000, 1e-8);
    double t1 = MPI_Wtime();

    // 4. max error: local max, then global max over all ranks
    double local_err = 0.0;
    for (std::size_t i = 0; i < x.size(); i++) {
        local_err = std::max(local_err, std::abs(x[i] - x_true[i]));
    }
    double global_err;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "ranks = " << size << ", n = " << n
                  << " (" << n * n << " unknowns)\n"
                  << "iterations = " << iters << "\n"
                  << "max error  = " << global_err << "\n"
                  << "time       = " << (t1 - t0) << " s\n";
    }

    MPI_Finalize();
    return 0;
}