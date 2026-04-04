// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#![allow(clippy::unit_arg)]

#[cfg(feature = "gpu-tests")]
mod bench_utils;

#[cfg(feature = "gpu-tests")]
use bench_utils::{make_gpu_sim, make_sparse_sim};
#[cfg(feature = "gpu-tests")]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
#[cfg(feature = "gpu-tests")]
use std::hint::black_box;

#[cfg(feature = "gpu-tests")]
fn single_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_qubit_gates");

    for num_qubits in [10, 15, 20, 25, 28] {
        let targets = [
            ("low", 0),
            ("mid", num_qubits / 2),
            ("high", num_qubits - 1),
        ];

        // --- Hadamard ---
        {
            let mut sim = make_gpu_sim(num_qubits);
            for q in 0..num_qubits {
                sim.h(q);
            }
            for &(pos_name, target) in &targets {
                group.bench_function(format!("H/{num_qubits}q/{pos_name}"), |b| {
                    b.iter(|| black_box(sim.h(target)));
                });
            }
        }

        // --- Pauli X ---
        {
            let mut sim = make_gpu_sim(num_qubits);
            for q in 0..num_qubits {
                sim.h(q);
            }
            for &(pos_name, target) in &targets {
                group.bench_function(format!("X/{num_qubits}q/{pos_name}"), |b| {
                    b.iter(|| black_box(sim.x(target)));
                });
            }
        }

        // --- T gate ---
        {
            let mut sim = make_gpu_sim(num_qubits);
            for q in 0..num_qubits {
                sim.h(q);
            }
            for &(pos_name, target) in &targets {
                group.bench_function(format!("T/{num_qubits}q/{pos_name}"), |b| {
                    b.iter(|| black_box(sim.t(target)));
                });
            }
        }

        // --- Rx(pi/4) ---
        {
            let mut sim = make_gpu_sim(num_qubits);
            for q in 0..num_qubits {
                sim.h(q);
            }
            let angle = std::f64::consts::FRAC_PI_4;
            for &(pos_name, target) in &targets {
                group.bench_function(format!("Rx_pi4/{num_qubits}q/{pos_name}"), |b| {
                    b.iter(|| black_box(sim.rx(angle, target)));
                });
            }
        }
    }
    group.finish();
}

#[cfg(feature = "gpu-tests")]
fn two_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("two_qubit_gates");

    for num_qubits in [10, 15, 20, 25] {
        // --- CNOT ---
        {
            let mut sim = make_gpu_sim(num_qubits);
            for q in 0..num_qubits {
                sim.h(q);
            }
            group.bench_function(format!("CNOT/{num_qubits}q/adjacent"), |b| {
                b.iter(|| black_box(sim.mcx(&[0], 1)));
            });
            group.bench_function(format!("CNOT/{num_qubits}q/distant"), |b| {
                b.iter(|| black_box(sim.mcx(&[0], num_qubits - 1)));
            });
        }

        // --- SWAP ---
        {
            let mut sim = make_gpu_sim(num_qubits);
            for q in 0..num_qubits {
                sim.h(q);
            }
            group.bench_function(format!("SWAP/{num_qubits}q/adjacent"), |b| {
                b.iter(|| black_box(sim.swap(0, 1)));
            });
            group.bench_function(format!("SWAP/{num_qubits}q/distant"), |b| {
                b.iter(|| black_box(sim.swap(0, num_qubits - 1)));
            });
        }
    }
    group.finish();
}

#[cfg(feature = "gpu-tests")]
fn multi_controlled_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_controlled_gates");

    for num_qubits in [10, 15, 20] {
        let mut sim = make_gpu_sim(num_qubits);
        for q in 0..num_qubits {
            sim.h(q);
        }

        for num_controls in [1, 2, 3] {
            let controls: Vec<usize> = (0..num_controls).collect();
            let target = num_controls;
            group.bench_function(format!("MCX/{num_controls}ctrl/{num_qubits}q"), |b| {
                b.iter(|| black_box(sim.mcx(&controls, target)));
            });
        }
    }
    group.finish();
}

#[cfg(feature = "gpu-tests")]
fn gpu_vs_sparse_single_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_vs_sparse/H_gate");

    for num_qubits in [10, 15, 20, 25] {
        // --- GPU ---
        {
            let mut gpu_sim = make_gpu_sim(num_qubits);
            for q in 0..num_qubits {
                gpu_sim.h(q);
            }
            group.bench_with_input(BenchmarkId::new("gpu", num_qubits), &num_qubits, |b, _| {
                b.iter(|| black_box(gpu_sim.h(0)));
            });
        }

        // --- Sparse ---
        {
            let (mut sparse_sim, ids) = make_sparse_sim(num_qubits);
            for &id in &ids {
                sparse_sim.h(id);
            }
            group.bench_with_input(
                BenchmarkId::new("sparse", num_qubits),
                &num_qubits,
                |b, _| {
                    b.iter(|| black_box(sparse_sim.h(ids[0])));
                },
            );
        }
    }
    group.finish();
}

#[cfg(feature = "gpu-tests")]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .warm_up_time(std::time::Duration::from_secs(3))
        .measurement_time(std::time::Duration::from_secs(5));
    targets = single_qubit_gates, two_qubit_gates, multi_controlled_gates, gpu_vs_sparse_single_gate
);

#[cfg(feature = "gpu-tests")]
criterion_main!(benches);

#[cfg(not(feature = "gpu-tests"))]
fn main() {
    eprintln!("bench 'single_gate' requires --features gpu-tests; skipping.");
}
