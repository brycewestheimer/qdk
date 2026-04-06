# GPU Simulator Benchmark Results

## Test Environment

- **GPU**: NVIDIA GeForce RTX 5070 (12 GB VRAM)
- **wgpu backend**: Vulkan
- **Driver**: 591.86
- **CPU**: Intel Core i7-14700K (20 cores, 28 threads)
- **RAM**: 128 GB DDR4
- **OS**: Windows 11
- **Rust**: 1.92.0 (edition 2024)
- **criterion**: 0.8.x
- **Date**: 2026-04-05

> **Note:** Some benchmarks for higher qubit counts (20q+) and many-shot throughput
> were skipped due to excessive run time (>10 min per configuration). These are
> flagged with **[SKIPPED]** in the tables below and can be collected in a future run.

## Crossover Point

For random Clifford+T circuits at depth >= 50:

- **< 12 qubits**: Sparse simulator is faster (GPU kernel launch overhead dominates)
- **12–16 qubits**: GPU and sparse are comparable at low depths; GPU wins at depth 500
- **> 16 qubits**: GPU simulator is significantly faster (parallelism dominates)

At 8 qubits, sparse is ~200x faster (1.2 ms vs 245 ms for depth 50). At 12 qubits
depth 500, performance is comparable (374 ms GPU vs 356 ms sparse). At 16 qubits
depth 500, GPU is ~23x faster (441 ms GPU vs 10.1 s sparse).

For QFT circuits:

- GPU overhead dominates at all tested sizes (4–16 qubits). Sparse is faster up to at
  least 16 qubits due to the relatively low gate count in QFT circuits. Crossover is
  expected above 16 qubits.

For chemistry (QPE) circuits:

- Sparse is faster at all tested configurations (8–12 total qubits). The chemistry
  circuits have relatively few qubits with moderate depth, keeping GPU overhead dominant.
  Crossover is expected above ~16 total qubits.

## Gate Time Scaling

Single gate application time (H gate, applied to qubit 0):

| Qubits | State vector size | H gate (GPU) | H gate (sparse) | Speedup   |
| ------ | ----------------- | ------------ | --------------- | --------- |
| 10     | 8 KB              | 60.4 µs      | 2.1 ns          | 0.00003x  |
| 15     | 256 KB            | 67.1 µs      | 2.1 ns          | 0.00003x  |
| 20     | 8 MB              | 71.3 µs      | 2.1 ns          | 0.00003x  |
| 25     | 256 MB            | 1.12 ms      | 2.1 ns          | 0.000002x |

> **Note:** The sparse simulator H gate benchmark reports ~2 ns because the sparse
> representation only touches non-zero amplitudes. For a single H gate on a mostly-|0⟩
> state, the sparse simulator does almost no work. The GPU simulator has fixed overhead
> per kernel dispatch regardless of state sparsity. The gpu_vs_sparse comparison is
> meaningful only for full circuits where the state becomes dense.
>
> 28-qubit benchmarks were skipped (GPU buffer limit: max 27 qubits on this GPU).

### Target Bit Position Effect

| Qubits | H gate (low bit) | H gate (mid bit) | H gate (high bit) |
| ------ | ---------------- | ---------------- | ----------------- |
| 10     | 62.4 µs          | 63.7 µs          | 62.8 µs           |
| 15     | 162.2 µs         | 165.3 µs         | 173.1 µs          |
| 20     | 173.4 µs         | 168.2 µs         | 171.2 µs          |
| 25     | 1.10 ms          | 2.06 ms          | 1.06 ms           |

At 10–20 qubits, target bit position has negligible effect (~5% variation).
At 25 qubits the mid-bit target is ~2x slower than low/high, likely due to
memory access pattern differences at large state vector sizes.

### Additional Gate Timings (GPU)

| Gate        | 10q      | 15q      | 20q      | 25q     |
| ----------- | -------- | -------- | -------- | ------- |
| X           | 64.1 µs  | 182.1 µs | 173.5 µs | 1.05 ms |
| T           | 165.2 µs | 790.3 µs | 136.1 µs | 1.04 ms |
| Rx(π/4)     | 162.6 µs | 358.7 µs | 136.0 µs | 1.04 ms |
| CNOT (adj)  | 60.0 µs  | 60.5 µs  | 66.1 µs  | 1.03 ms |
| CNOT (dist) | 60.0 µs  | 59.9 µs  | 86.0 µs  | 1.04 ms |
| SWAP (adj)  | 64.8 µs  | 69.0 µs  | 134.1 µs | 1.10 ms |
| SWAP (dist) | 65.0 µs  | 65.2 µs  | 93.0 µs  | 1.06 ms |
| MCX 1-ctrl  | 61.8 µs  | 70.1 µs  | 121.0 µs | —       |
| MCX 2-ctrl  | 63.1 µs  | 63.8 µs  | 117.3 µs | —       |
| MCX 3-ctrl  | 70.8 µs  | 63.8 µs  | 66.7 µs  | —       |

## Circuit Benchmark Results

### Random Clifford+T Circuits

| Qubits | Depth 50 (GPU) | Depth 50 (sparse) | Depth 100 (GPU) | Depth 100 (sparse) | Depth 500 (GPU) | Depth 500 (sparse) |
| ------ | -------------- | ----------------- | --------------- | ------------------ | --------------- | ------------------ |
| 8      | 244.6 ms       | 1.22 ms           | 248.9 ms        | 2.85 ms            | 321.4 ms        | 15.3 ms            |
| 12     | 275.7 ms       | 23.3 ms           | 246.2 ms        | 62.2 ms            | 373.5 ms        | 356.0 ms           |
| 16     | 230.9 ms       | 679.4 ms          | 289.5 ms        | 1.71 s             | 441.1 ms        | 10.13 s            |
| 20     | 227.8 ms       | **[SKIPPED]**     | **[SKIPPED]**   | **[SKIPPED]**      | **[SKIPPED]**   | **[SKIPPED]**      |
| 24     | **[SKIPPED]**  | **[SKIPPED]**     | **[SKIPPED]**   | **[SKIPPED]**      | **[SKIPPED]**   | **[SKIPPED]**      |

Key observations:

- At **8 qubits**, sparse is ~200x faster (GPU overhead dominates).
- At **12 qubits depth 500**, GPU and sparse converge (374 ms vs 356 ms).
- At **16 qubits**, GPU is **2.9x faster** at depth 50, **5.9x faster** at depth 100,
  and **23x faster** at depth 500.

### QFT Scaling

| Qubits | GPU           | Sparse        | Speedup  |
| ------ | ------------- | ------------- | -------- |
| 4      | 197.6 ms      | 6.1 µs        | 0.00003x |
| 8      | 200.1 ms      | 53.2 µs       | 0.0003x  |
| 12     | 202.3 ms      | 588.7 µs      | 0.003x   |
| 16     | 210.0 ms      | 11.1 ms       | 0.053x   |
| 20     | **[SKIPPED]** | **[SKIPPED]** | —        |

GPU overhead (~200 ms baseline) dominates at these qubit counts. The QFT circuit
has O(n²) gates, so sparse scales well up to 16 qubits. At 16 qubits sparse
takes 11 ms — GPU would need ~50+ qubits before the parallelism benefit
outweighs the fixed dispatch overhead.

### Chemistry QPE Circuits

| Configuration                  | GPU      | Sparse   | Speedup |
| ------------------------------ | -------- | -------- | ------- |
| 4sys + 4anc, 10 Trotter steps  | 200.1 ms | 53.1 µs  | 0.0003x |
| 4sys + 4anc, 50 Trotter steps  | 207.8 ms | 317.0 µs | 0.002x  |
| 4sys + 4anc, 100 Trotter steps | 226.5 ms | 650.5 µs | 0.003x  |
| 4sys + 8anc, 10 Trotter steps  | 200.4 ms | 53.0 µs  | 0.0003x |
| 4sys + 8anc, 50 Trotter steps  | 210.2 ms | 332.9 µs | 0.002x  |
| 4sys + 8anc, 100 Trotter steps | 224.8 ms | 623.9 µs | 0.003x  |

At 8–12 total qubits, sparse is dramatically faster. The GPU's ~200 ms fixed
overhead dominates. For larger molecular simulations (20+ qubits), the
crossover point would favor GPU.

### Many-Shot Throughput

| Qubits | 100 shots (GPU) | 100 shots (sparse) | 1000 shots (GPU) | 1000 shots (sparse) |
| ------ | --------------- | ------------------ | ---------------- | ------------------- |
| 8      | 20.8 s          | 113.8 ms           | **[SKIPPED]**    | **[SKIPPED]**       |
| 12     | **[SKIPPED]**   | **[SKIPPED]**      | **[SKIPPED]**    | **[SKIPPED]**       |
| 16     | **[SKIPPED]**   | **[SKIPPED]**      | **[SKIPPED]**    | **[SKIPPED]**       |

> **Note:** Many-shot benchmarks were extremely slow. At 8 qubits / 100 shots,
> the GPU took 20.8 s (vs 114 ms sparse). The 1000-shot configuration estimated
> ~34 minutes for a single benchmark and was killed. The many-shot benchmark runs
> the full circuit + measurement pipeline per shot, so GPU dispatch overhead
> compounds across shots. This benchmark warrants redesign — batching shots on the
> GPU (running the circuit once and sampling multiple times) would be more efficient.

## Measurement Pipeline Cost

| Qubits | Measure single qubit | get_state() readback | Joint measure (1q) | Joint measure (2q) | Joint measure (4q) |
| ------ | -------------------- | -------------------- | ------------------ | ------------------ | ------------------ |
| 10     | 99.8 µs              | 76.6 µs              | 101.7 µs           | 127.3 µs           | 171.2 µs           |
| 15     | 112.7 µs             | 1.55 ms              | 99.5 µs            | 130.7 µs           | 174.8 µs           |
| 20     | **[SKIPPED]**        | **[SKIPPED]**        | **[SKIPPED]**      | **[SKIPPED]**      | **[SKIPPED]**      |
| 25     | **[SKIPPED]**        | **[SKIPPED]**        | **[SKIPPED]**      | **[SKIPPED]**      | **[SKIPPED]**      |

Key observations:

- Single-qubit measurement takes ~100–113 µs regardless of qubit count (10–15q).
- `get_state()` readback scales with state vector size: 77 µs at 10q (8 KB) vs
  1.55 ms at 15q (256 KB).
- Joint measurement cost scales linearly with number of measured qubits
  (~30 µs per additional measured qubit).

## f32 Precision Characteristics

Fidelity of GPU (f32) vs sparse (f64) reference for random universal circuits:

| Qubits | Depth 10    | Depth 50    | Depth 100     | Depth 500     | Depth 1000    | Depth 5000    |
| ------ | ----------- | ----------- | ------------- | ------------- | ------------- | ------------- |
| 4      | 0.999999709 | 0.999998148 | 0.999996076   | 0.999983366   | 0.999963293   | 0.999826170   |
| 8      | 0.999999285 | 0.999996969 | 0.999993285   | 0.999966531   | 0.999927644   | 0.999665915   |
| 12     | 0.999999328 | 0.999993791 | 0.999989583   | 0.999947297   | 0.999897210   | 0.999493991   |
| 16     | 0.999998790 | 0.999994392 | 0.999987241   | 0.999935534   | 0.999862737   | 0.999325793   |
| 20     | 0.999998131 | 0.999993232 | **[SKIPPED]** | **[SKIPPED]** | **[SKIPPED]** | **[SKIPPED]** |

### Detailed Precision Metrics

| Qubits | Depth | Max Error | RMS Error | Fidelity    | Trace Distance |
| ------ | ----- | --------- | --------- | ----------- | -------------- |
| 4      | 10    | 1.16e-7   | 5.05e-8   | 0.999999709 | 5.39e-4        |
| 4      | 100   | 9.80e-7   | 5.11e-7   | 0.999996076 | 1.98e-3        |
| 4      | 1000  | 7.81e-6   | 4.62e-6   | 0.999963293 | 6.06e-3        |
| 4      | 5000  | 3.22e-5   | 2.18e-5   | 0.999826170 | 1.32e-2        |
| 8      | 100   | 5.09e-7   | 2.17e-7   | 0.999993285 | 2.59e-3        |
| 8      | 1000  | 5.01e-6   | 2.27e-6   | 0.999927644 | 8.51e-3        |
| 8      | 5000  | 2.38e-5   | 1.05e-5   | 0.999665915 | 1.83e-2        |
| 12     | 100   | 2.47e-7   | 8.31e-8   | 0.999989583 | 3.23e-3        |
| 12     | 1000  | 2.50e-6   | 8.05e-7   | 0.999897210 | 1.01e-2        |
| 12     | 5000  | 1.14e-5   | 3.96e-6   | 0.999493991 | 2.25e-2        |
| 16     | 100   | 9.79e-6   | 4.59e-8   | 0.999987241 | 3.57e-3        |
| 16     | 1000  | 8.87e-7   | 2.69e-7   | 0.999862737 | 1.17e-2        |
| 16     | 5000  | 4.48e-6   | 1.32e-6   | 0.999325793 | 2.60e-2        |

### Precision Summary

| Circuit depth | Max qubits at F > 0.999 | Max qubits at F > 0.99 |
| ------------- | ----------------------- | ---------------------- |
| 10            | 20+                     | 20+                    |
| 100           | 20+                     | 20+                    |
| 1000          | 16+                     | 20+                    |
| 5000          | 4                       | 16+                    |

**Key finding:** f32 precision remains excellent (F > 0.999) for all tested
configurations up to depth 1000. Even at depth 5000, fidelity stays above 0.999
for up to 4 qubits and above 0.99 for all tested sizes. For typical chemistry
QPE circuits (depth < 1000, < 20 qubits), f32 precision is more than sufficient.

Error grows approximately as O(depth × ε) where ε ≈ 1e-7 (f32 machine epsilon).

## When to Use Each Simulator

| Scenario                     | Recommendation    | Rationale                                           |
| ---------------------------- | ----------------- | --------------------------------------------------- |
| < 16 qubits, any circuit     | Sparse            | GPU ~200 ms fixed overhead not amortized            |
| >= 16 qubits, dense circuits | GPU               | 6–23x speedup at depth 100–500                      |
| 12–16 qubits                 | Depends           | Benchmark your specific circuit                     |
| High precision required      | Sparse (f64)      | f32 error accumulates at depth > 5000               |
| Chemistry QPE (< 16q)        | Sparse            | GPU overhead dominates at small qubit counts        |
| Chemistry QPE (>= 16q)       | GPU               | Dense states + many Trotter steps favor parallelism |
| Circuit depth > 5000         | Sparse or GPU+f64 | f32 fidelity drops below 0.999                      |
| No GPU available             | Sparse            | GPU sim requires wgpu-compatible hardware           |

## Benchmarks Flagged for Future Collection

The following benchmarks were skipped because a single configuration exceeded
10 minutes of run time:

- **Random circuit scaling**: 20q and 24q at all depths (dense sparse sim at 20q+ is very slow)
- **QFT scaling**: 20q (sparse reference sim too slow)
- **Many-shot throughput**: All 1000-shot configs; 12q and 16q at 100 shots
  (GPU dispatch-per-shot compounds; consider batched sampling)
- **Measurement pipeline**: 20q and 25q configurations
- **Precision report**: 20q at depths 100, 500, 1000, 5000
  (sparse reference computation at 20q × depth 5000 is prohibitively slow)
