#!/usr/bin/env bash
# Benchmark the hybrid MPI+OpenMP CG solver for several ranks x threads splits.
# Each configuration runs RUNS times; the median time is reported.
#
# Usage: ./bench/run_bench.sh [n] [runs]
#   n     grid size (n x n unknowns), default 1000
#   runs  repetitions per configuration, default 5
set -euo pipefail

N=${1:-1000}
RUNS=${2:-5}
EXE=./build/cg
OUT=bench/results.csv

if [ ! -x "$EXE" ]; then
    echo "Build first: cmake --build build -j" >&2
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

# ranks x threads configurations, grouped by total cores
CONFIGS="1x1 2x1 1x2 4x1 2x2 1x4"
if [ "$CORES" -ge 8 ]; then
    CONFIGS="$CONFIGS 8x1 4x2 2x4 1x8"
fi

echo "ranks,threads,cores,run,time_s,iterations" > "$OUT"
printf "%-8s %-6s %-12s %-10s %s\n" "config" "cores" "median (s)" "speedup" "iterations"

BASE=""
for cfg in $CONFIGS; do
    R=${cfg%x*}
    T=${cfg#*x}
    times=""
    iters=""
    for run in $(seq 1 "$RUNS"); do
        out=$(mpirun --bind-to none -x OMP_NUM_THREADS="$T" -np "$R" "$EXE" "$N")
        t=$(echo "$out" | awk '/^time/ {print $3}')
        iters=$(echo "$out" | awk '/^iterations/ {print $3}')
        echo "$R,$T,$((R * T)),$run,$t,$iters" >> "$OUT"
        times="$times $t"
    done
    med=$(echo $times | tr ' ' '\n' | sort -g | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
    if [ -z "$BASE" ]; then BASE=$med; fi
    sp=$(awk -v b="$BASE" -v m="$med" 'BEGIN {printf "%.2fx", b/m}')
    printf "%-8s %-6s %-12s %-10s %s\n" "${R}x${T}" "$((R * T))" "$med" "$sp" "$iters"
done

echo
echo "Raw results written to $OUT"
