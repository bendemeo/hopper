# Changelog

## 0.2.0 (2026-03-21)

### Added
- `pyproject.toml` with proper packaging metadata and optional dependency groups
- Comprehensive test suite (pytest) covering hopper, treehopper, and partitions
- Type annotations and numpy-style docstrings throughout
- GitHub Actions CI running tests on push and PR
- Examples directory with runnable quickstart script
- MIT license file

### Changed
- Rewrote README with installation instructions, API reference, and citation
- Replaced `print()` calls with `logging` module
- Lazy-import scanpy in explore module (now truly optional at runtime)

### Fixed
- Bug in `treehopper.get_wts()` referencing class instead of instance

### Removed
- Stale notebook with embedded outputs (replaced by `examples/`)
- Checked-in `.pyc` files and scratch files

## 0.1.0 (2019)

Initial research implementation.

### Added
- `hopper` class: greedy k-centers via farthest-first traversal
- `treehopper` class: scalable variant with pre-partitioning
- `PCATreePartition` and `RPartition` partition strategies
- `explore` module with scanpy/AnnData visualization utilities
- `haus` and `haus_curve` for evaluating sketch quality
- Voronoi cell tracking and weight computation
