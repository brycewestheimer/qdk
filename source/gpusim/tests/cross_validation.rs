#![cfg(feature = "gpu-tests")]

mod cross_validation {
    mod circuit;
    mod generator;
    mod runners;

    mod edge_case_tests;
    mod fundamental_tests;
    mod random_tests;
    mod structured_tests;
}
