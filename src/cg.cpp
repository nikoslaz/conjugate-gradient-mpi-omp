#include "cg.hpp"
#include "vec_ops.hpp"
#include "halo.hpp"

int cg(const CSR& A, const Partition& part, const std::vector<double>& b, std::vector<double>& x, int max_iter, double tol) {
    std::vector<double> r = b;
    std::vector<double> p = r;
    std::vector<double> q(part.local_n);
    std::vector<double> p_ext;
    double rr = dot(r, r);
    x.assign(part.local_n, 0.0);
    double norm_b = std::sqrt(dot(b,b));
    
    int i = 0;
    for (; i < max_iter; i++) {
        halo_exchange(part, p, p_ext);
        spmv(A, p_ext, q);
        double alpha = rr/ dot(p,q);
        axpy(alpha, p, x);
        axpy(-alpha, q, r);
        double rr_new = dot(r, r);

        if (std::sqrt(rr_new) < tol * norm_b) { return i+1; }

        double beta = rr_new / rr;
        const int len = static_cast<int>(p.size());
        #pragma omp parallel for
        for (int j = 0; j < len; j++) {
            p[j] = r[j] + beta * p[j];
        }

        rr = rr_new;
    }

    return i;
}