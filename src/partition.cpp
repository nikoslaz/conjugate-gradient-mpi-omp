#include "partition.hpp"
#include <algorithm>
#include <mpi.h>

Partition fill_part(int n, int rank, int size) {
    Partition part;
    int rem = n % size;

    part.base = n / size;
    part.first_grid_row = rank * part.base + std::min(rank, rem);
    part.n = n;

    if (rank < rem) { part.local_grid_rows = part.base+1; }
    else            { part.local_grid_rows = part.base; }

    part.local_n = part.local_grid_rows * n;
    part.global_offset = part.first_grid_row * n;

    if (rank == 0) { part.up = MPI_PROC_NULL; }
    else           { part.up = rank-1; }

    if (rank == size - 1) { part.down = MPI_PROC_NULL; }
    else                  { part.down = rank+1; }

    return part;
}