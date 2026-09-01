# Contributing

## Development workflow

1. Create an issue or define the experiment question.
2. Create a focused branch.
3. Add or modify one logical experiment.
4. Build for the target CUDA architecture.
5. Run correctness tests.
6. Run Compute Sanitizer on a real GPU.
7. Capture benchmark evidence when making performance claims.
8. Open a pull request with commands and results.

Keep commits small and descriptive.

Recommended commit prefixes:

- `feat:` new experiment/capability
- `fix:` correctness bug
- `perf:` measured performance improvement
- `test:` validation coverage
- `docs:` documentation
- `build:` CMake/CI/toolchain
- `chore:` repository maintenance

## New experiment checklist

- [ ] Clear learning objective/hypothesis
- [ ] CMake target
- [ ] Deterministic input
- [ ] Correctness oracle
- [ ] Numeric error metric
- [ ] CTest entry
- [ ] CUDA Event timing
- [ ] Meaningful throughput metric
- [ ] Compute Sanitizer clean
- [ ] Documentation updated

## Style

C++/CUDA code follows the repository `.clang-format` file.

Prefer:

- RAII where practical;
- explicit error checking for every CUDA Runtime call;
- kernel launch error checks;
- bounded, deterministic tests;
- no hidden global state;
- separate correctness and performance problem sizes.

## Performance PRs

A performance PR must contain both baseline and new results from the same hardware/software environment.

Include:

- GPU model;
- compute capability;
- driver;
- CUDA Toolkit;
- OS;
- command line;
- problem size;
- iteration count;
- before/after metric;
- confirmation that correctness and sanitizers pass.

## Licensing

The repository does not yet declare an open-source license. Do not copy third-party sample code into the repository. Implement experiments independently and cite conceptual references in documentation.
