#!/usr/bin/env bash
# Benchmark the hybrid MPI+OpenMP CG solver for several ranks x threads splits.
# Each configuration runs RUNS times; the summary reports the median time.
#
# Usage:
#   ./bench/run_bench.sh [n] [runs]          run all configurations (fresh results.csv)
#   CONFIGS="1x8" ./bench/run_bench.sh       re-run only the listed configurations;
#                                            their old rows in results.csv are replaced,
#                                            all other rows are kept
#   ./bench/run_bench.sh --summary           print the table from results.csv, no runs
#   ./bench/run_bench.sh --sizes [runs]      problem-size sweep (n = 250..2000) with the
#                                            best split (2 ranks x 4 threads) -> sizes.csv
#
#   n     grid size (n x n unknowns), default 1000
#   runs  repetitions per configuration, default 5
set -euo pipefail

OUT=bench/results.csv
EXE=./build/cg

# Estimated memory traffic per CG iteration and per unknown (see README): ~190 bytes.
BYTES_PER_UNKNOWN=190

summary() {
    # Median time per configuration, speedup relative to 1x1, in the order they appear.
    # N (grid size) is needed for the bandwidth estimate; default 1000.
    awk -F, -v n="${N:-1000}" -v bpu="$BYTES_PER_UNKNOWN" 'NR > 1 {
        key = $1 "x" $2
        if (!(key in seen)) { seen[key] = 1; order[++k] = key; cores[key] = $3 }
        cnt[key]++; t[key, cnt[key]] = $5; it[key] = $6
    }
    END {
        printf "%-8s %-6s %-6s %-12s %-10s %-11s %-10s %s\n", "config", "cores", "runs", "median (s)", "speedup", "ms/iter", "~GB/s", "iterations"
        for (j = 1; j <= k; j++) {
            key = order[j]; c = cnt[key]
            # insertion sort of this configuration'"'"'s times
            for (a = 1; a <= c; a++) v[a] = t[key, a] + 0
            for (a = 2; a <= c; a++) { x = v[a]; b = a - 1
                while (b >= 1 && v[b] > x) { v[b+1] = v[b]; b-- }
                v[b+1] = x }
            med[key] = (c % 2) ? v[(c+1)/2] : (v[c/2] + v[c/2+1]) / 2
        }
        base = ("1x1" in med) ? med["1x1"] : med[order[1]]
        for (j = 1; j <= k; j++) {
            key = order[j]
            ms = 1000 * med[key] / it[key]
            gbs = (bpu * n * n) / (ms / 1000) / 1e9
            printf "%-8s %-6s %-6s %-12.3f %-10s %-11.2f %-10.0f %s\n", key, cores[key], cnt[key], med[key], sprintf("%.2fx", base / med[key]), ms, gbs, it[key]
        }
    }' "$OUT"
}

if [ "${1:-}" = "--summary" ]; then
    summary
    exit 0
fi

if [ "${1:-}" = "--sizes" ]; then
    # Problem-size sweep with a fixed split: shows iterations ~ n and time ~ n^3.
    RUNS=${2:-3}
    SOUT=bench/sizes.csv
    R=2; T=4
    if [ ! -x "$EXE" ]; then echo "Build first" >&2; exit 1; fi
    echo "Problem-size sweep, ${R} ranks x ${T} threads, $RUNS runs per size"
    echo "n,unknowns,run,time_s,iterations" > "$SOUT"
    for n in 250 500 1000 2000; do
        for run in $(seq 1 "$RUNS"); do
            out=$(mpirun --bind-to none -x OMP_NUM_THREADS="$T" -np "$R" "$EXE" "$n")
            t=$(echo "$out" | awk '/^time/ {print $3}')
            iters=$(echo "$out" | awk '/^iterations/ {print $3}')
            echo "$n,$((n * n)),$run,$t,$iters" >> "$SOUT"
            echo "  n=$n run $run: $t s ($iters iterations)"
        done
    done
    echo
    awk -F, -v bpu="$BYTES_PER_UNKNOWN" 'NR > 1 {
        if (!($1 in seen)) { seen[$1] = 1; order[++k] = $1; u[$1] = $2 }
        c[$1]++; t[$1, c[$1]] = $4 + 0; it[$1] = $5
    } END {
        printf "%-6s %-10s %-11s %-12s %-10s %s\n", "n", "unknowns", "iterations", "median (s)", "ms/iter", "~GB/s"
        for (j = 1; j <= k; j++) {
            key = order[j]; m = c[key]
            for (a = 1; a <= m; a++) v[a] = t[key, a]
            for (a = 2; a <= m; a++) { x = v[a]; b = a - 1
                while (b >= 1 && v[b] > x) { v[b+1] = v[b]; b-- }
                v[b+1] = x }
            med = (m % 2) ? v[(m+1)/2] : (v[m/2] + v[m/2+1]) / 2
            ms = 1000 * med / it[key]
            printf "%-6s %-10s %-11s %-12.3f %-10.3f %.0f\n", key, u[key], it[key], med, ms, bpu * u[key] / (ms / 1000) / 1e9
        }
    }' "$SOUT"
    echo
    echo "Raw results written to $SOUT"
    exit 0
fi

N=${1:-1000}
RUNS=${2:-5}

if [ ! -x "$EXE" ]; then
    echo "Build first: cmake -S . -B build && cmake --build build -j" >&2
    exit 1
fi

# Machine info (macOS)
CORES=$(sysctl -n hw.physicalcpu)
PCORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo "?")
ECORES=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo "?")
CPU=$(sysctl -n machdep.cpu.brand_string)
echo "Machine: $CPU, $CORES cores ($PCORES performance + $ECORES efficiency)"
echo "Grid: n = $N ($((N * N)) unknowns), $RUNS runs per config"
echo

if [ -n "${CONFIGS:-}" ]; then
    # Partial re-run: keep results.csv, drop only the rows of the configs we re-run.
    if [ -f "$OUT" ]; then
        awk -F, -v list=" $CONFIGS " 'NR == 1 || index(list, " " $1 "x" $2 " ") == 0' "$OUT" > "$OUT.tmp"
        mv "$OUT.tmp" "$OUT"
    else
        echo "ranks,threads,cores,run,time_s,iterations" > "$OUT"
    fi
else
    CONFIGS="1x1 2x1 1x2 4x1 2x2 1x4"
    if [ "$CORES" -ge 8 ]; then
        CONFIGS="$CONFIGS 8x1 4x2 2x4 1x8"
    fi
    echo "ranks,threads,cores,run,time_s,iterations" > "$OUT"
fi

for cfg in $CONFIGS; do
    R=${cfg%x*}
    T=${cfg#*x}
    for run in $(seq 1 "$RUNS"); do
        out=$(mpirun --bind-to none -x OMP_NUM_THREADS="$T" -np "$R" "$EXE" "$N")
        t=$(echo "$out" | awk '/^time/ {print $3}')
        iters=$(echo "$out" | awk '/^iterations/ {print $3}')
        echo "$R,$T,$((R * T)),$run,$t,$iters" >> "$OUT"
        echo "  ${R}x${T} run $run: $t s ($iters iterations)"
    done
done

# Keep rows grouped by configuration in a fixed order (re-runs are appended at the end).
{
    head -n 1 "$OUT"
    tail -n +2 "$OUT" | awk -F, '{ printf "%d %d %s\n", $3, -$1, $0 }' | sort -k1,1n -k2,2n -k3 | cut -d' ' -f3-
} > "$OUT.tmp"
mv "$OUT.tmp" "$OUT"

echo
summary
echo
echo "Raw results written to $OUT"
