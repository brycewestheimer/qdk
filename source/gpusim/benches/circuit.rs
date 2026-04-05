// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#![allow(clippy::unit_arg)]

#[cfg(feature = "gpu-tests")]
mod bench_utils;

#[cfg(feature = "gpu-tests")]
use bench_utils::{
    apply_circuit_gpu, apply_circuit_sparse, chemistry_qpe_circuit, make_gpu_sim, make_sparse_sim,
    qft_circuit, random_clifford_t_circuit,
};
#[cfg(feature = "gpu-tests")]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
#[cfg(feature = "gpu-tests")]
use std::hint::black_box;

#[cfg(feature = "gpu-tests")]
fn random_circuit_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_circuit_scaling");
    group.measurement_time(std::time::Duration::from_secs(10));

    for num_qubits in [8, 12, 16, 20, 24] {
        for depth in [50, 100, 500] {
            let circuit = random_clifford_t_circuit(num_qubits, depth);

            // GPU
            group.bench_with_input(
                BenchmarkId::new("gpu", format!("{num_qubits}q_d{depth}")),
                &(num_qubits, depth),
                |b, _| {
                    b.iter(|| {
                        let mut sim = make_gpu_sim(num_qubits);
                        apply_circuit_gpu(&mut sim, &circuit);
                        sim.sync_gpu();
                        black_box(&sim);
                    });
                },
            );

            // Sparse
            group.bench_with_input(
                BenchmarkId::new("sparse", format!("{num_qubits}q_d{depth}")),
                &(num_qubits, depth),
                |b, _| {
                    b.iter(|| {
                        let (mut sim, ids) = make_sparse_sim(num_qubits);
                        apply_circuit_sparse(&mut sim, &ids, &circuit);
                        black_box(&sim);
                    });
                },
            );
        }
    }
    group.finish();
}

#[cfg(feature = "gpu-tests")]
fn qft_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("qft_scaling");

    for num_qubits in [4, 8, 12, 16, 20] {
        let circuit = qft_circuit(num_qubits);

        group.bench_with_input(BenchmarkId::new("gpu", num_qubits), &num_qubits, |b, _| {
            b.iter(|| {
                let mut sim = make_gpu_sim(num_qubits);
                apply_circuit_gpu(&mut sim, &circuit);
                sim.sync_gpu();
                black_box(&sim);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("sparse", num_qubits),
            &num_qubits,
            |b, _| {
                b.iter(|| {
                    let (mut sim, ids) = make_sparse_sim(num_qubits);
                    apply_circuit_sparse(&mut sim, &ids, &circuit);
                    black_box(&sim);
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "gpu-tests")]
fn chemistry_circuit(c: &mut Criterion) {
    let mut group = c.benchmark_group("chemistry_circuit");
    group.measurement_time(std::time::Duration::from_secs(15));

    let system_qubits = 4;
    for ancilla_qubits in [4, 8] {
        for trotter_steps in [10, 50, 100] {
            let circuit = chemistry_qpe_circuit(system_qubits, ancilla_qubits, trotter_steps);
            let total = system_qubits + ancilla_qubits;
            let label = format!("{system_qubits}sys_{ancilla_qubits}anc_{trotter_steps}trotter");

            // GPU
            group.bench_function(format!("gpu/{label}"), |b| {
                b.iter(|| {
                    let mut sim = make_gpu_sim(total);
                    apply_circuit_gpu(&mut sim, &circuit);
                    sim.sync_gpu();
                    black_box(&sim);
                });
            });

            // Sparse
            group.bench_function(format!("sparse/{label}"), |b| {
                b.iter(|| {
                    let (mut sim, ids) = make_sparse_sim(total);
                    apply_circuit_sparse(&mut sim, &ids, &circuit);
                    black_box(&sim);
                });
            });
        }
    }
    group.finish();
}

#[cfg(feature = "gpu-tests")]
fn many_shot_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("many_shot_throughput");
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);

    for num_qubits in [8, 12, 16] {
        let circuit = random_clifford_t_circuit(num_qubits, 50);

        for num_shots in [100u64, 1000] {
            // GPU
            group.bench_function(format!("gpu/{num_qubits}q/{num_shots}shots"), |b| {
                b.iter(|| {
                    for shot in 0..num_shots {
                        let mut sim = make_gpu_sim(num_qubits);
                        sim.set_rng_seed(shot);
                        apply_circuit_gpu(&mut sim, &circuit);
                        black_box(sim.measure(0).expect("measurement should succeed"));
                    }
                });
            });

            // Sparse
            group.bench_function(format!("sparse/{num_qubits}q/{num_shots}shots"), |b| {
                b.iter(|| {
                    for _shot in 0..num_shots {
                        let (mut sim, ids) = make_sparse_sim(num_qubits);
                        apply_circuit_sparse(&mut sim, &ids, &circuit);
                        black_box(sim.measure(ids[0]));
                    }
                });
            });
        }
    }
    group.finish();
}

#[cfg(feature = "gpu-tests")]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(std::time::Duration::from_secs(5));
    targets = random_circuit_scaling, qft_scaling, chemistry_circuit, many_shot_throughput
);

#[cfg(feature = "gpu-tests")]
criterion_main!(benches);

#[cfg(not(feature = "gpu-tests"))]
fn main() {
    eprintln!("bench 'circuit' requires --features gpu-tests; skipping.");
}
