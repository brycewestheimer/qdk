// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#![allow(clippy::unit_arg)]

#[cfg(feature = "gpu-tests")]
mod bench_utils;

#[cfg(feature = "gpu-tests")]
use bench_utils::make_gpu_sim;
#[cfg(feature = "gpu-tests")]
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(feature = "gpu-tests")]
use rand::{Rng, SeedableRng, rngs::StdRng};
#[cfg(feature = "gpu-tests")]
use std::hint::black_box;

#[cfg(feature = "gpu-tests")]
fn measurement_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("measurement_time");

    for num_qubits in [10, 15, 20, 25] {
        let mut sim = make_gpu_sim(num_qubits);
        let mut rng = StdRng::seed_from_u64(0xBE0C_CAFE);
        for q in 0..num_qubits {
            sim.h(q);
        }
        for q in 0..num_qubits {
            if rng.gen_bool(0.5) {
                sim.t(q);
            }
        }

        // Measure single qubit (probability + collapse)
        group.bench_function(format!("measure_single/{num_qubits}q"), |b| {
            b.iter(|| {
                sim.h(0);
                black_box(sim.measure(0).expect("measurement should succeed"))
            });
        });

        // Full state readback
        group.bench_function(format!("get_state/{num_qubits}q"), |b| {
            b.iter(|| black_box(sim.get_state().expect("get_state should succeed")));
        });
    }
    group.finish();
}

#[cfg(feature = "gpu-tests")]
fn joint_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("joint_measurement");

    for num_qubits in [10, 15, 20] {
        let mut sim = make_gpu_sim(num_qubits);
        for q in 0..num_qubits {
            sim.h(q);
        }

        for num_measured in [1, 2, 4] {
            let qubits: Vec<usize> = (0..num_measured).collect();
            group.bench_function(format!("joint/{num_measured}q_of_{num_qubits}q"), |b| {
                b.iter(|| {
                    for &q in &qubits {
                        sim.h(q);
                    }
                    black_box(
                        sim.joint_measure(&qubits)
                            .expect("joint measurement should succeed"),
                    )
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
        .sample_size(50)
        .warm_up_time(std::time::Duration::from_secs(3))
        .measurement_time(std::time::Duration::from_secs(5));
    targets = measurement_time, joint_measurement
);

#[cfg(feature = "gpu-tests")]
criterion_main!(benches);

#[cfg(not(feature = "gpu-tests"))]
fn main() {
    eprintln!("bench 'measurement' requires --features gpu-tests; skipping.");
}
