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
  run_bench.sh      benchmark script (ranks × threads sweep, problem-size sweep)
  results.csv       raw results of the ranks × threads sweep
  sizes.csv         raw results of the problem-size sweep
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

Setup: Apple M4 laptop (10 cores: 4 performance + 6 efficiency), Open MPI, Release build, `tol = 1e-8`. Each configuration was run 5 times and the median is reported. Raw data: [`bench/results.csv`](bench/results.csv) and [`bench/sizes.csv`](bench/sizes.csv).

To reproduce:

```bash
./bench/run_bench.sh 1000 5              # all ranks × threads configurations, 5 runs each
./bench/run_bench.sh --sizes 3           # problem-size sweep with 2 ranks × 4 threads
CONFIGS="1x8" ./bench/run_bench.sh       # re-run only some configurations (others are kept)
./bench/run_bench.sh --summary           # print the table from results.csv without running
```

### Ranks × threads (n = 1000, 10⁶ unknowns)

| ranks × threads | cores | median time (s) | speedup | ms / iteration | est. bandwidth (GB/s) |
|---|---|---|---|---|---|
| 1 × 1 | 1 | 6.68 | 1.00× | 3.89 | 49 |
| 2 × 1 | 2 | 4.11 | 1.62× | 2.40 | 79 |
| 1 × 2 | 2 | 4.18 | 1.60× | 2.44 | 78 |
| 4 × 1 | 4 | 3.31 | 2.01× | 1.93 | 98 |
| 2 × 2 | 4 | 3.29 | 2.03× | 1.92 | 99 |
| 1 × 4 | 4 | 3.31 | 2.02× | 1.93 | 98 |
| 8 × 1 | 8 | 3.22 | 2.07× | 1.88 | 101 |
| 4 × 2 | 8 | 3.12 | 2.14× | 1.82 | 104 |
| 2 × 4 | 8 | **3.05** | **2.19×** | **1.78** | **107** |
| 1 × 8 | 8 | 3.20 | 2.08× | 1.87 | 102 |

Every run converged in exactly **1715 iterations** with a maximum error of about 2.25 · 10⁻⁷, so all parallel versions compute the same thing as the serial one. Run-to-run variation was small (typically under 2%).

The bandwidth column is an estimate: about 190 bytes of memory traffic per unknown per iteration (see below), divided by the measured time per iteration.

### Analysis: CG is memory-bound

The speedup reaches 2× at 4 cores and then almost stops. The reason is **memory bandwidth**, not communication:

- The halo exchange moves only 2 × 1000 doubles (16 KB) per iteration, which is negligible.
- One CG iteration streams roughly **190 MB** through memory at n = 1000: the CSR arrays (`values` 40 MB, `col_idx` 20 MB, `row_ptr` 4 MB), plus about ten passes over 8 MB vectors (SpMV input/output, two dot products, two axpys, the `p` update and the halo copy).
- A single core reaches about **49 GB/s**. With 4 cores the estimate is already about **98 GB/s**, and the best configuration reaches about **107 GB/s**, close to the M4's quoted memory bandwidth of 120 GB/s.

SpMV does about 2 floating-point operations per 12 bytes loaded, an arithmetic intensity of roughly 0.17 flop/byte. In roofline terms it sits far on the memory-bound side: once the cores together saturate the memory bus, more cores (ranks or threads) cannot make it faster.

This explains the results:

- **At 4 cores, the split does not matter**: 4 × 1, 2 × 2 and 1 × 4 all take 3.3 s. Ranks and threads share the same memory system and hit the same limit.
- **Going from 4 to 8 cores adds only ~8%.** The M4 has only 4 performance cores, so the extra 4 are efficiency cores, and the memory bus is already nearly saturated.
- **The hybrid split is slightly best** at 8 cores (2 × 4 at 3.05 s vs 3.22 s for pure MPI and 3.20 s for pure OpenMP). Pure MPI does more halo copies and has more participants in each `Allreduce`; pure OpenMP has a larger fork/join and reduction cost per loop.

On a real cluster, the picture changes: each node brings its own memory bandwidth, so adding nodes scales the total bandwidth, and the hybrid model (one or a few ranks per node or socket, threads inside) reduces the number of MPI messages and the cost of the global reductions.

### Problem size (2 ranks × 4 threads)

| n | unknowns | iterations | median time (s) | ms / iteration | est. bandwidth (GB/s) |
|---|---|---|---|---|---|
| 250 | 62,500 | 444 | 0.077 | 0.17 | 68 |
| 500 | 250,000 | 873 | 0.321 | 0.37 | 129 |
| 1000 | 1,000,000 | 1715 | 2.97 | 1.73 | 110 |
| 2000 | 4,000,000 | 3361 | 26.9 | 8.00 | 95 |

- **Iterations grow linearly with n** (444 → 873 → 1715 → 3361, roughly doubling each time). This matches theory: CG needs O(√κ) iterations, and the condition number κ of the Poisson matrix grows like n².
- **Total time grows like n³**: doubling n gives 4× more unknowns and 2× more iterations, so about 8× more time (measured: 9.3× from 500 to 1000, 9.0× from 1000 to 2000).
- **n = 500 appears faster than memory** (129 GB/s, above the 120 GB/s peak). Its working set (~48 MB) is small enough that part of it is served from the caches.
- **n = 250 is slower per byte** (68 GB/s). Each iteration takes only 0.17 ms, so the fixed costs per iteration (three synchronisation points, OpenMP fork/join for each loop, MPI latency) become a large share of the time.
- **n = 2000 drops to 95 GB/s**: the working set (~760 MB) is far larger than any cache, so everything comes from main memory.

## Limitations and possible extensions

- Benchmarks were run on a single laptop, not a cluster, so inter-node scaling is not measured.
- 1D (row-block) partition only. A 2D block partition would reduce halo volume at large rank counts.
- No preconditioner. A Jacobi or multigrid preconditioner would cut the iteration count.
- The halo exchange is blocking. Non-blocking `MPI_Isend`/`MPI_Irecv` could overlap communication with the interior part of SpMV.
- Pipelined CG variants merge the two `Allreduce` calls into one per iteration.
