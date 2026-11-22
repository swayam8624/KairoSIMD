# Teaching source map — KairoSIMD
**Deep source walkthrough:** [CPU SIMD dispatch and numerical parity](SIMD_SCALAR_PARITY_GUIDE.md). Continue with [the advanced module, dispatch and numerical-contract lab](ADVANCED_SIMD_API_AND_DISPATCH_LAB.md).

Portable SIMD math kernels with scalar fallbacks.

Read these source links in order, identify the caller and callee for each public API, then check the error/ownership contract and its tests.

## Reading order

1. [`README.md`](../README.md)
2. [`CMakeLists.txt`](../CMakeLists.txt)
3. [`src`](../src)
4. [`tests`](../tests)
5. [`research`](../research)

## Repository integration and API contracts

MLLibrary/Tensor and Kairo compute callers rely on numerical equivalence between vectorized and scalar paths. Verify CPU feature dispatch, alignment and tolerance assumptions.

## Teaching standards

- Comments explain invariants, decision reasons, lifetime, prerequisites and cross-file effects; they do not narrate obvious statements.
- Follow upstream interfaces before implementation details; distinguish implemented, host-verified and planned features.
- Run available tests at a pinned commit and consult [STATUS.md](../STATUS.md) / [STATUS.yaml](../STATUS.yaml) before declaring completion.
- Reuse original contributions under [LICENSE](../LICENSE), subject to third-party code and asset terms.


## Complete source-file inventory

After the curated teaching path, follow this deterministic inventory of first-party source, scripts, shaders, and tests. Grouping is a suggested learning order, not an indication that every function is already annotated. Vendored dependencies, ROMs, binaries, data, media, and generated evidence are excluded.

### Entry points and build (1)

1. [`CMakeLists.txt`](../CMakeLists.txt)

### Interfaces and modules (1)

1. [`src/KairoSIMD.cppm`](../src/KairoSIMD.cppm)

### Implementation and other source (1)

1. [`src/KairoSIMDPlatform.cpp`](../src/KairoSIMDPlatform.cpp)

### Verification and tests (2)

1. [`tests/benchmark.cpp`](../tests/benchmark.cpp)
2. [`tests/smoke.cpp`](../tests/smoke.cpp)

Snapshot totals: 5 first-party source/build/test files, 13 tracked files overall. Re-audit when files are renamed or added.
