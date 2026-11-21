# Teaching source map — KairoSIMD

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
