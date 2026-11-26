#include "halo.hpp"
#include <mpi.h>

void halo_exchange(const Partition& part, const std::vector<double>& p, std::vector<double>& p_ext) {
    const int n       = part.n;
    const int local_n = part.local_n;

    // make room: [halo above (n) | own (local_n) | halo below (n)]
    p_ext.assign(local_n + 2 * n, 0.0);

    // opy my own values into the middle
    std::copy(p.begin(), p.end(), p_ext.begin() + n);

    // send my FIRST row up, receive into my halo BELOW from down
    MPI_Sendrecv(p.data(),                   n, MPI_DOUBLE, part.up,   0,
                 p_ext.data() + n + local_n, n, MPI_DOUBLE, part.down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // send my LAST row down, receive into my halo ABOVE from up
    MPI_Sendrecv(p.data() + local_n - n,     n, MPI_DOUBLE, part.down, 0,
                 p_ext.data(),               n, MPI_DOUBLE, part.up,   0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}