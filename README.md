# conjugate-gradient-mpi-omp

A distributed **Conjugate Gradient (CG)** solver for the 2D Poisson equation, written in C++20 with **hybrid MPI + OpenMP** parallelism.

The sparse matrix is stored in CSR format and split across MPI ranks by grid rows. Each CG iteration does one halo exchange (`MPI_Sendrecv`) before the sparse matrix-vector product and two `MPI_Allreduce` calls for the dot products. Inside each rank, the loops are parallelised with OpenMP.

## The problem

We solve **A x = b**, where A is the 5-point finite-difference discretisation of the 2D Poisson equation on an `n × n` grid:

- one unknown per grid point, so A has `n²` rows
- each row has at most 5 non-zeros: `4` on the diagonal and `-1` for each of the up/left/right/down neighbours
- A is symmetric positive definite, which is what CG requires

For `n = 1000` that is 1,000,000 unknowns and about 5 million non-zeros. The matrix is generated directly in CSR format (`row_ptr`, `col_idx`, `values`), so the dense matrix is never stored.

**Verification:** the program sets `x_true = 1`, computes `b = A · x_true`, solves from `x = 0`, and reports the maximum error `max |x − x_true|`.

## The algorithm

```
x = 0,  r = b,  p = r,  rr = r·r
loop:
    q      = A·p                 halo exchange + SpMV
    alpha  = rr / (p·q)          dot  → MPI_Allreduce
    x      = x + alpha·p         axpy (local)
    r      = r − alpha·q         axpy (local)
    rr_new = r·r                 dot  → MPI_Allreduce
    stop if ‖r‖ < tol · ‖b‖
    p      = r + (rr_new/rr)·p   local
    rr     = rr_new
```

Each iteration therefore communicates **three times**: one neighbour-to-neighbour halo exchange and two global reductions.

## Parallelisation

### MPI: row-block partition by grid rows

Each rank owns a contiguous block of **whole grid rows** (`n / P` rows, with the remainder spread over the first ranks). With this split, each rank needs exactly one grid row (`n` values) from the rank above and one from the rank below.

### Halo exchange

Before each SpMV, `p` is copied into an extended vector:

```
p_ext = [ halo above (n) | own values (local_n) | halo below (n) ]
```

Two `MPI_Sendrecv` calls fill the halos: everyone sends their first row up, then everyone sends their last row down. Because every rank sends in the same direction in each call, there is no deadlock. At the domain edges the neighbour is `MPI_PROC_NULL`, so no special cases are needed.

The local CSR matrix stores column indices that already point into `p_ext` (`local column = global column − global_offset + n`), so the SpMV kernel is the same in the serial and the distributed versions.

### Dot products

Each rank computes a local partial sum and `MPI_Allreduce(MPI_SUM)` gives every rank the global result. `axpy` needs no communication.

### OpenMP

The SpMV row loop, the dot-product loop (`reduction(+:sum)`), the `axpy` loops and the `p` update use `#pragma omp parallel for`. MPI is initialised with `MPI_THREAD_FUNNELED`: only the main thread makes MPI calls, outside the parallel regions.

## Project structure

```
include/            headers
  partition.hpp     which grid rows each rank owns, neighbour ranks
  csr.hpp           CSR matrix + local Poisson matrix builder
  halo.hpp          halo exchange
  spmv.hpp          sparse matrix-vector product
  vec_ops.hpp       distributed dot, axpy
  cg.hpp            the CG solver
src/                implementations + main.cpp
bench/
  run_bench.sh      benchmark script (ranks × threads sweep)
  results.csv       raw benchmark results
```

## Build

Requirements: a C++20 compiler, CMake ≥ 3.16, an MPI implementation (e.g. Open MPI) and OpenMP. On macOS, install the OpenMP runtime with `brew install libomp`. CMake finds it automatically.

```bash
cmake -S . -B build          # defaults to a Release (-O3) build
cmake --build build -j
```

## Run

```bash
# ranks × threads: 2 MPI ranks, 4 OpenMP threads each, 1000 × 1000 grid
mpirun --bind-to none -x OMP_NUM_THREADS=4 -np 2 ./build/cg 1000
```

```
ranks = 2, n = 1000 (1000000 unknowns)
iterations = 1715
max error  = 2.25418e-07
time       = 3.05 s
```

`--bind-to none` matters for hybrid runs: by default Open MPI binds each rank to one core, so all of that rank's OpenMP threads would share a single core.

## Results

Setup: `n = 1000` (10⁶ unknowns), `tol = 1e-8`, Release build, Open MPI, on a single Apple Silicon laptop. Each configuration was run 5 times (the `1 × 8` configuration 3 times) and the median is reported. Raw data: [`bench/results.csv`](bench/results.csv).

| ranks × threads | cores | median time (s) | speedup vs 1×1 |
|---|---|---|---|
| 1 × 1 | 1 | 6.89 | 1.00× |
| 2 × 1 | 2 | 4.11 | 1.68× |
| 1 × 2 | 2 | 4.15 | 1.66× |
| 4 × 1 | 4 | 3.67 | 1.88× |
| 2 × 2 | 4 | 3.65 | 1.89× |
| 1 × 4 | 4 | 3.95 | 1.74× |
| 8 × 1 | 8 | 3.71 | 1.86× |
| 4 × 2 | 8 | 3.16 | 2.18× |
| 2 × 4 | 8 | **3.06** | **2.25×** |
| 1 × 8 | 8 | 3.33 | 2.07× |

Every configuration converged in exactly **1715 iterations** with a maximum error of about 2.25 · 10⁻⁷, so the parallel versions compute the same thing as the serial one.

### Analysis: CG is memory-bound

The speedup stops at about 2× even with 8 cores. The reason is **memory bandwidth**, not communication:

- The halo exchange moves only 2 × 1000 doubles (16 KB) per iteration, which is negligible.
- One CG iteration streams roughly **190 MB** through memory: the CSR arrays (`values` 40 MB, `col_idx` 20 MB, `row_ptr` 4 MB), plus about ten passes over 8 MB vectors (SpMV input/output, two dot products, two axpys, the `p` update and the halo copy).
- At 4.0 ms per iteration (1 × 1), that is about **47 GB/s** from a single core. At 1.8 ms per iteration (2 × 4), about **105 GB/s**, which is close to the laptop's total memory bandwidth.

SpMV does about 2 floating-point operations per 12 bytes loaded, so its arithmetic intensity is very low (~0.17 flop/byte). In roofline terms, it sits far on the memory-bound side: once the cores together saturate the memory bus, adding more cores (ranks or threads) cannot make it faster.

This also explains why **the split between ranks and threads matters little here**: on one machine, ranks and threads share the same memory system, so they hit the same limit. The hybrid configurations (`2 × 4`, `4 × 2`) are slightly faster than pure MPI (`8 × 1`), which has more halo copies and more participants in each `Allreduce`, and slightly faster than pure OpenMP (`1 × 8`).

On a real cluster, the picture changes: each node brings its own memory bandwidth, so adding nodes scales the bandwidth, and the hybrid model (one or a few ranks per node or socket, threads inside) reduces the number of MPI messages and the cost of the global reductions.

### Convergence

The number of iterations grows linearly with the grid size: 183 for `n = 100`, 873 for `n = 500`, 1715 for `n = 1000`. This matches theory: CG needs O(√κ) iterations, and the condition number κ of the Poisson matrix grows like n².

## Limitations and possible extensions

- Benchmarks were run on a single laptop, not a cluster, so inter-node scaling is not measured.
- 1D (row-block) partition only. A 2D block partition would reduce halo volume at large rank counts.
- No preconditioner. A Jacobi or multigrid preconditioner would cut the iteration count.
- The halo exchange is blocking. Non-blocking `MPI_Isend`/`MPI_Irecv` could overlap communication with the interior part of SpMV.
- Pipelined CG variants merge the two `Allreduce` calls into one per iteration.
