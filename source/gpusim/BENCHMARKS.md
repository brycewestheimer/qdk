# GPU Simulator Benchmark Results

## Test Environment

- **GPU**: [e.g., NVIDIA RTX 4090 (16384 CUDA cores, 24 GB VRAM)]
- **wgpu backend**: [e.g., Vulkan 1.3]
- **Driver**: [e.g., NVIDIA 555.42.02]
- **CPU**: [e.g., AMD Ryzen 9 7950X (16 cores, 32 threads)]
- **RAM**: [e.g., 64 GB DDR5-5600]
- **OS**: [e.g., Ubuntu 24.04 LTS / Windows 11 WSL2]
- **Rust**: [e.g., 1.82.0 (edition 2024)]
- **criterion**: 0.8.x
- **Date**: [YYYY-MM-DD]

## Crossover Point

For random Clifford+T circuits at depth >= 50:

- **< [X] qubits**: Sparse simulator is faster (GPU kernel launch overhead dominates)
- **[X]-[Y] qubits**: Comparable performance (crossover region)
- **> [Y] qubits**: GPU simulator is faster (parallelism dominates)

For QFT circuits:

- Crossover occurs at approximately [X]-[Y] qubits.

For chemistry (QPE) circuits:

- Crossover occurs at approximately [X]-[Y] qubits.

## Gate Time Scaling

| Qubits | State vector size | H gate (GPU) | H gate (sparse) | Speedup |
| ------ | ----------------- | ------------ | --------------- | ------- |
| 10     | 8 KB              |              |                 |         |
| 15     | 256 KB            |              |                 |         |
| 20     | 8 MB              |              |                 |         |
| 25     | 256 MB            |              |                 |         |
| 28     | 2 GB              |              |                 |         |

### Target Bit Position Effect

| Qubits | H gate (low bit) | H gate (mid bit) | H gate (high bit) |
| ------ | ---------------- | ---------------- | ----------------- |
| 20     |                  |                  |                   |
| 25     |                  |                  |                   |
| 28     |                  |                  |                   |

## Circuit Benchmark Results

### Random Clifford+T Circuits

| Qubits | Depth 50 (GPU) | Depth 50 (sparse) | Depth 100 (GPU) | Depth 100 (sparse) | Depth 500 (GPU) | Depth 500 (sparse) |
| ------ | -------------- | ----------------- | --------------- | ------------------ | --------------- | ------------------ |
| 8      |                |                   |                 |                    |                 |                    |
| 12     |                |                   |                 |                    |                 |                    |
| 16     |                |                   |                 |                    |                 |                    |
| 20     |                |                   |                 |                    |                 |                    |
| 24     |                |                   |                 |                    |                 |                    |

### QFT Scaling

| Qubits | GPU | Sparse | Speedup |
| ------ | --- | ------ | ------- |
| 4      |     |        |         |
| 8      |     |        |         |
| 12     |     |        |         |
| 16     |     |        |         |
| 20     |     |        |         |

### Chemistry QPE Circuits

| Configuration                  | GPU | Sparse | Speedup |
| ------------------------------ | --- | ------ | ------- |
| 4sys + 4anc, 10 Trotter steps  |     |        |         |
| 4sys + 4anc, 50 Trotter steps  |     |        |         |
| 4sys + 4anc, 100 Trotter steps |     |        |         |
| 4sys + 8anc, 10 Trotter steps  |     |        |         |
| 4sys + 8anc, 50 Trotter steps  |     |        |         |
| 4sys + 8anc, 100 Trotter steps |     |        |         |

### Many-Shot Throughput

| Qubits | 100 shots (GPU) | 100 shots (sparse) | 1000 shots (GPU) | 1000 shots (sparse) |
| ------ | --------------- | ------------------ | ---------------- | ------------------- |
| 8      |                 |                    |                  |                     |
| 12     |                 |                    |                  |                     |
| 16     |                 |                    |                  |                     |

## Measurement Pipeline Cost

| Qubits | Measure single qubit | get_state() readback | Joint measure (2q) | Joint measure (4q) |
| ------ | -------------------- | -------------------- | ------------------ | ------------------ |
| 10     |                      |                      |                    |                    |
| 15     |                      |                      |                    |                    |
| 20     |                      |                      |                    |                    |
| 25     |                      |                      |                    |                    |

## f32 Precision Characteristics

| Qubits | Depth 10 | Depth 50 | Depth 100 | Depth 500 | Depth 1000 | Depth 5000 |
| ------ | -------- | -------- | --------- | --------- | ---------- | ---------- |
| 4      |          |          |           |           |            |            |
| 8      |          |          |           |           |            |            |
| 12     |          |          |           |           |            |            |
| 16     |          |          |           |           |            |            |
| 20     |          |          |           |           |            |            |

(Each cell: fidelity F, with max_error and trace_distance in the full report)

### Precision Summary

| Circuit depth | Max qubits at F > 0.999 | Max qubits at F > 0.99 |
| ------------- | ----------------------- | ---------------------- |
| 10            |                         |                        |
| 100           |                         |                        |
| 1000          |                         |                        |
| 5000          |                         |                        |

## When to Use Each Simulator

| Scenario                     | Recommendation    | Rationale                                 |
| ---------------------------- | ----------------- | ----------------------------------------- |
| < [X] qubits, any circuit    | Sparse            | GPU overhead not amortized                |
| > [Y] qubits, dense circuits | GPU               | Significant speedup                       |
| [X]-[Y] qubits               | Depends           | Benchmark your specific circuit           |
| Many shots (> 100)           | GPU               | Throughput advantage compounds            |
| High precision required      | Sparse (f64)      | f32 error accumulates at depth > [Z]      |
| Chemistry QPE                | GPU               | Target use case; dense states, many shots |
| Circuit depth > [Z]          | Sparse or GPU+f64 | f32 precision may be insufficient         |
| No GPU available             | Sparse            | GPU sim requires wgpu-compatible hardware |
